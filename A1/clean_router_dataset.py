"""
Data cleaning cho Router dataset (A1.2 follow-up) — rà soát lại nhãn (intent) của
toàn bộ router_dataset.jsonl bằng chính LLM giáo viên, để sửa các mẫu bị gán nhãn
sai (phát hiện qua sample_mistakes của lần fine-tune đầu — xem ví dụ như
"i wanna order a double espresso" bị gán "consultant" thay vì "order").

Thiết kế giữ đúng phong cách hạ tầng đã dùng ở A1.2 (generate_router_data.py):
  - Async API calls (AsyncOpenAI + asyncio.Semaphore).
  - Batch nhiều câu / 1 lần gọi để tiết kiệm chi phí + thời gian (20 câu/batch).
  - Checkpoint/resume: đọc file review đã có, chỉ xử lý phần còn thiếu.
  - Progress tracking bằng tqdm.
  - Retry với exponential backoff.

Cách chạy (dùng đúng 3 biến môi trường như A1.2):
    $env:OPENAI_API_KEY="..."
    $env:OPENAI_BASE_URL="..."      (nếu dùng proxy/OpenAI-compatible endpoint)
    $env:GEN_MODEL="gpt-4o-mini"    (hoặc model bạn đã dùng để sinh data)
    python clean_router_dataset.py --input router_dataset.jsonl

Output:
  - router_dataset_review.jsonl   : kết quả rà soát từng mẫu (index, text, nhãn cũ,
                                     nhãn LLM đề xuất, độ tin cậy).
  - router_dataset_cleaned.jsonl  : dataset đã áp dụng sửa nhãn (chỉ sửa khi LLM
                                     phản đối nhãn cũ với confidence "high" hoặc
                                     "medium" — giữ nguyên nếu "low" để tránh
                                     áp đặt sai lên các câu THẬT SỰ mơ hồ).
  - Bảng tổng kết in ra terminal: bao nhiêu mẫu bị đổi, đổi theo cặp nhãn nào,
    tập trung ở hard hay normal.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import Counter

from openai import AsyncOpenAI
from tqdm import tqdm

MODEL_NAME = os.getenv("GEN_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

MAX_CONCURRENCY = 1  # local llama-cpp-python server (CPU-only) chỉ xử lý tuần tự -> tránh "Connection error"
BATCH_SIZE = 20
MAX_RETRIES = 3
BASE_BACKOFF = 1.5

LABELS = ["order", "consultant", "faq", "ignore"]
LABEL_TO_ID = {lbl: i for i, lbl in enumerate(LABELS)}

# Định nghĩa taxonomy — GIỮ NGUYÊN VĂN với system prompt production (router_agent_fast.py /
# finetune_router.py) để việc rà soát nhãn nhất quán với đúng cách Router sẽ được chấm điểm.
TAXONOMY = """Định nghĩa 4 nhãn ý định (intent) của khách hàng tại quán cà phê Highlands Coffee:
- "order": Đặt hàng, tính tiền, gọi thêm/bớt món — hành động CỤ THỂ liên quan đến việc mua/gọi món.
  Ví dụ: "Cho 1 ly bạc xỉu size M", "Tính tiền giúp em", "I'd like a large iced latte", "I wanna order a double espresso".
- "consultant": Nhờ TƯ VẤN, xin GỢI Ý món (chưa chốt gọi món cụ thể) — khách đang phân vân, hỏi ý kiến.
  Ví dụ: "Có món nào ngọt mà không quá béo không?", "What do you recommend for a hot day?".
- "faq": Hỏi thông tin CỦA QUÁN (wifi, giờ mở cửa, địa chỉ, chỗ gửi xe, chính sách...). Chỉ chọn faq khi
  câu hỏi thật sự hỏi về quán, không phải câu hỏi phiếm.
  Ví dụ: "Wifi pass gì vậy em?", "What time do you close?".
- "ignore": Câu nói KHÔNG mang yêu cầu/câu hỏi thật sự nào tới quán — chào hỏi vu vơ, test mic/kết nối,
  tiếng ồn, cười đùa, than vãn không liên quan, gọi tên suông. Đây KHÔNG phải faq dù có dạng câu hỏi,
  và KHÔNG phải order/consultant nếu không có hành động/yêu cầu cụ thể nào về đồ uống.
  Ví dụ: "Alo alo test mic", "haha", "Sao lâu rồi không thấy anh tới vậy?"."""


def parse_json_array(content: str):
    content = content.strip()
    content = re.sub(r"^```(json)?", "", content).strip()
    content = re.sub(r"```$", "", content).strip()
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        content = match.group(0)
    return json.loads(content)


def build_judge_messages(batch: list[dict]) -> list[dict]:
    items_text = "\n".join(
        f'{i}. (nhãn hiện tại: {rec["intent"]}) "{rec["text"]}"' for i, rec in enumerate(batch)
    )
    system = (
        "Bạn là chuyên gia gán nhãn dữ liệu (data annotator) cực kỳ cẩn thận, khách quan. "
        "Nhiệm vụ: RÀ SOÁT lại nhãn ý định đã gán cho từng câu — nhãn hiện tại có thể ĐÚNG hoặc SAI, "
        "bạn phải tự đánh giá độc lập dựa trên định nghĩa, không mặc định tin nhãn hiện tại."
    )
    user = (
        f"{TAXONOMY}\n\n"
        "Hãy xác định nhãn ĐÚNG NHẤT cho từng câu dưới đây theo đúng 4 định nghĩa trên "
        "(chọn nhãn phản ánh Ý ĐỊNH CHÍNH, RÕ RÀNG NHẤT của câu, kể cả khi câu có thể mơ hồ giữa 2 nhãn):\n\n"
        f"{items_text}\n\n"
        'Trả về DUY NHẤT một JSON array, mỗi phần tử dạng: '
        '{"index": <số thứ tự ở trên>, "label": "<order|consultant|faq|ignore>", "confidence": "<high|medium|low>"}. '
        "KHÔNG kèm giải thích, KHÔNG markdown, KHÔNG text thừa ngoài JSON array."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def judge_batch(batch: list[dict], semaphore: asyncio.Semaphore):
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=build_judge_messages(batch),
                    temperature=0,
                )
                content = resp.choices[0].message.content
                parsed = parse_json_array(content)
                return parsed
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"\n[Lỗi] batch thất bại sau {MAX_RETRIES} lần thử: {e}")
                    return None
                await asyncio.sleep(BASE_BACKOFF * (2 ** attempt))
    return None


def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_existing_review(path: str) -> dict[int, dict]:
    if not os.path.exists(path):
        return {}
    reviewed = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                reviewed[rec["index"]] = rec
    return reviewed


async def main(input_path: str, review_path: str, cleaned_path: str, only_hard: bool = False):
    if not API_KEY:
        raise SystemExit("Thiếu OPENAI_API_KEY — set biến môi trường trước khi chạy (giống A1.2).")

    records = load_dataset(input_path)
    print(f"Đã load {len(records)} mẫu từ {input_path}")

    reviewed = load_existing_review(review_path)
    print(f"Đã có {len(reviewed)} mẫu được review từ trước (checkpoint/resume)")

    if only_hard:
        eligible = [i for i, r in enumerate(records) if r.get("is_hard")]
        print(f"Chế độ --only-hard: chỉ rà soát {len(eligible)}/{len(records)} mẫu hard "
              f"(mẫu normal giữ nguyên, không đụng tới).")
    else:
        eligible = list(range(len(records)))

    todo_indices = [i for i in eligible if i not in reviewed]
    batches = [todo_indices[i:i + BATCH_SIZE] for i in range(0, len(todo_indices), BATCH_SIZE)]

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    review_f = open(review_path, "a", encoding="utf-8")

    pbar = tqdm(total=len(todo_indices), desc="Rà soát nhãn", unit="mẫu")

    async def process_batch(idx_group: list[int]):
        batch_records = [records[i] for i in idx_group]
        result = await judge_batch(batch_records, semaphore)
        if result is None:
            pbar.update(len(idx_group))
            return
        for item in result:
            local_i = item.get("index")
            if local_i is None or not (0 <= local_i < len(idx_group)):
                continue
            global_i = idx_group[local_i]
            rec = records[global_i]
            review_rec = {
                "index": global_i,
                "text": rec["text"],
                "original_label": rec["intent"],
                "judged_label": item.get("label"),
                "confidence": item.get("confidence", "low"),
                "is_hard": rec.get("is_hard", False),
                "language": rec.get("language"),
            }
            review_f.write(json.dumps(review_rec, ensure_ascii=False) + "\n")
        review_f.flush()
        pbar.update(len(idx_group))

    for i in range(0, len(batches), MAX_CONCURRENCY):
        chunk = batches[i:i + MAX_CONCURRENCY]
        await asyncio.gather(*[process_batch(b) for b in chunk])

    pbar.close()
    review_f.close()

    # ==================== Áp dụng sửa nhãn ====================
    reviewed = load_existing_review(review_path)
    changed = []
    flagged_low_confidence = []
    cleaned_records = []
    for i, rec in enumerate(records):
        new_rec = dict(rec)
        rv = reviewed.get(i)
        if rv and rv["judged_label"] in LABEL_TO_ID and rv["judged_label"] != rv["original_label"]:
            if rv["confidence"] in ("high", "medium"):
                new_rec["intent"] = rv["judged_label"]
                new_rec["label"] = LABEL_TO_ID[rv["judged_label"]]
                changed.append(rv)
            else:
                flagged_low_confidence.append(rv)
        cleaned_records.append(new_rec)

    with open(cleaned_path, "w", encoding="utf-8") as f:
        for rec in cleaned_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ==================== Báo cáo tổng kết ====================
    print("\n" + "=" * 60)
    print("BÁO CÁO LÀM SẠCH DATASET")
    print("=" * 60)
    print(f"Tổng số mẫu             : {len(records)}")
    print(f"Số mẫu bị đổi nhãn      : {len(changed)} ({len(changed) / len(records) * 100:.2f}%)")
    print(f"  - trong đó hard       : {sum(1 for c in changed if c['is_hard'])}")
    print(f"  - trong đó normal     : {sum(1 for c in changed if not c['is_hard'])}")
    print(f"Số mẫu nghi ngờ nhưng LLM confidence=low (GIỮ NGUYÊN, không tự sửa): {len(flagged_low_confidence)}")

    pair_counter = Counter((c["original_label"], c["judged_label"]) for c in changed)
    print("\nCác cặp đổi nhãn phổ biến (nhãn cũ -> nhãn mới):")
    for (old, new), cnt in pair_counter.most_common():
        print(f"  {old:<12} -> {new:<12} : {cnt}")

    print("\nPhân bố nhãn SAU khi làm sạch:")
    new_dist = Counter(r["intent"] for r in cleaned_records)
    for lbl in LABELS:
        print(f"  {lbl:<12}: {new_dist[lbl]}")

    print(f"\nĐã lưu:\n  - Chi tiết review : {review_path}\n  - Dataset đã sạch : {cleaned_path}")
    if flagged_low_confidence:
        low_conf_path = review_path.replace(".jsonl", "_low_confidence.jsonl")
        with open(low_conf_path, "w", encoding="utf-8") as f:
            for c in flagged_low_confidence:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"  - Mẫu nghi ngờ (confidence thấp, cần xem thủ công nếu muốn): {low_conf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="router_dataset.jsonl")
    parser.add_argument("--review-output", default="router_dataset_review.jsonl")
    parser.add_argument("--cleaned-output", default="router_dataset_cleaned.jsonl")
    parser.add_argument("--only-hard", action="store_true",
                         help="Chỉ rà soát ~600 hard samples thay vì toàn bộ 4000 mẫu (nhanh hơn nhiều trên CPU).")
    args = parser.parse_args()
    asyncio.run(main(args.input, args.review_output, args.cleaned_output, only_hard=args.only_hard))