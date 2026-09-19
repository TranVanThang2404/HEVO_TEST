
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import re
import json
import random
import asyncio
from collections import defaultdict

from openai import AsyncOpenAI
from tqdm import tqdm

# ==================== CONFIG ====================

MODEL_NAME = os.getenv("GEN_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")  # None -> dùng OpenAI mặc định

OUTPUT_FILE = "router_dataset.jsonl"

LANGUAGES = ["vi", "en"]

INTENTS = [
    {
        "label": 0,
        "intent": "order",
        "desc_vi": "Đặt hàng, gọi món, tính tiền, thêm/bớt món, chỉnh size/đường/đá.",
        "desc_en": "Placing an order, asking for the bill, adding/removing items, adjusting size/sugar/ice.",
    },
    {
        "label": 1,
        "intent": "consultant",
        "desc_vi": "Nhờ tư vấn, xin gợi ý món dựa trên khẩu vị/ngân sách/thời tiết, hỏi món nào ngon/bán chạy.",
        "desc_en": "Asking for recommendations based on taste/budget/weather, asking what's good or popular.",
    },
    {
        "label": 2,
        "intent": "faq",
        "desc_vi": "Hỏi thông tin CHUNG về quán: wifi, giờ mở/đóng cửa, địa chỉ, chỗ gửi xe, chính sách, khuyến mãi.",
        "desc_en": "Asking general info about the shop: wifi, opening hours, address, parking, policies, promotions.",
    },
    {
        "label": 3,
        "intent": "ignore",
        "desc_vi": "Câu nói KHÔNG mang yêu cầu/câu hỏi cụ thể tới quán: chào hỏi vu vơ, test mic/kết nối, tiếng ồn, cười đùa, gọi tên suông, câu vô nghĩa.",
        "desc_en": "Utterances that carry no real request to the shop: idle greetings, mic/connection tests, noise, laughter, calling out a name, nonsense filler.",
    },
]

HARD_PAIRS = {
    "order": "consultant",       # "Có gì ngon rẻ không, cho em 1 ly" - vừa hỏi vừa order
    "consultant": "order",
    "faq": "ignore",             # "Nghe rõ không đó" - dạng câu hỏi nhưng thực chất là test mic (ignore)
    "ignore": "faq",
}

TOTAL_TARGET = 4000
PER_CLASS_TARGET = TOTAL_TARGET // len(INTENTS)    
HARD_RATIO = 0.15                                    
PER_CLASS_HARD = int(PER_CLASS_TARGET * HARD_RATIO)  
PER_CLASS_NORMAL = PER_CLASS_TARGET - PER_CLASS_HARD  
# Chia đều cho 2 ngôn ngữ
PER_CELL_NORMAL = PER_CLASS_NORMAL // len(LANGUAGES)  
PER_CELL_HARD = PER_CLASS_HARD // len(LANGUAGES)      

BATCH_SIZE = 15
MAX_CONCURRENCY = 5
MAX_RETRIES = 3
BASE_BACKOFF = 1.5  

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)



def build_messages(intent_cfg: dict, language: str, is_hard: bool, batch_size: int):
    intent = intent_cfg["intent"]
    desc = intent_cfg["desc_vi"] if language == "vi" else intent_cfg["desc_en"]
    lang_name = "tiếng Việt" if language == "vi" else "tiếng Anh"

    if is_hard:
        confusable = HARD_PAIRS[intent]
        confusable_cfg = next(i for i in INTENTS if i["intent"] == confusable)
        confusable_desc = confusable_cfg["desc_vi"] if language == "vi" else confusable_cfg["desc_en"]
        task = (
            f"Hãy tạo {batch_size} câu nói KHÓ / MƠ HỒ bằng {lang_name} mà khách hàng quán cà phê có thể nói. "
            f"Mỗi câu phải trông giống CẢ hai ý định sau, nhưng ý định ĐÚNG cuối cùng phải là '{intent}':\n"
            f"- '{intent}': {desc}\n"
            f"- (dễ nhầm với) '{confusable}': {confusable_desc}\n"
            f"Câu phải TỰ NHIÊN như người thật nói, có thể viết tắt, thiếu dấu, khẩu ngữ. "
            f"Đây là các câu KHÓ dùng để kiểm tra khả năng phân biệt ranh giới giữa 2 ý định."
        )
    else:
        task = (
            f"Hãy tạo {batch_size} câu nói ĐA DẠNG bằng {lang_name} mà khách hàng quán cà phê Highlands Coffee "
            f"có thể nói với ý định '{intent}': {desc}\n"
            f"Đa dạng về độ dài, cách xưng hô, mức độ trang trọng/thân mật, có thể có lỗi chính tả nhẹ hoặc viết tắt "
            f"như người thật nhắn tin/nói chuyện. KHÔNG lặp lại cấu trúc câu giống nhau."
        )

    system = (
        "Bạn là công cụ sinh dữ liệu huấn luyện cho một hệ thống AI phân loại ý định khách hàng quán cà phê. "
        "Luôn trả lời DUY NHẤT một mảng JSON hợp lệ chứa các chuỗi (list of strings), không thêm giải thích, "
        "không thêm markdown code fence, không đánh số thứ tự."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]


def parse_json_array(raw_text: str):
    """Cố gắng parse output của model thành list[str], chịu được vài lỗi định dạng nhẹ."""
    text = raw_text.strip()
    # bỏ code fence nếu có
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except json.JSONDecodeError:
        pass
    # fallback: cố tìm mảng JSON đầu tiên trong text bằng regex
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return []


def normalize_for_dedup(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)  # bỏ dấu câu để bắt các câu gần trùng
    return t


# ==================== GENERATION ====================

async def generate_batch(intent_cfg: dict, language: str, is_hard: bool, batch_size: int):
    """Gọi API sinh 1 batch câu, có retry + exponential backoff. Trả về list[str] (có thể rỗng nếu fail hết)."""
    messages = build_messages(intent_cfg, language, is_hard, batch_size)

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=1.0, 
                )
            raw = response.choices[0].message.content
            items = parse_json_array(raw)
            items = [s for s in items if 2 <= len(s) <= 200]
            return items
        except Exception as e:
            wait = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"[Retry {attempt + 1}/{MAX_RETRIES}] lỗi khi gọi API ({intent_cfg['intent']}, {language}, "
                  f"hard={is_hard}): {e} -> chờ {wait:.1f}s")
            await asyncio.sleep(wait)

    print(f"[Bỏ qua] hết số lần retry cho ({intent_cfg['intent']}, {language}, hard={is_hard})")
    return []


def load_existing():
    """Đọc file output đã có (nếu chạy dở lần trước) để resume, tránh sinh lại từ đầu."""
    records = []
    seen = set()
    counts = defaultdict(int)  

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(rec)
                seen.add(normalize_for_dedup(rec["text"]))
                key = (rec["intent"], rec["language"], rec.get("is_hard", False))
                counts[key] += 1

    return records, seen, counts


async def fill_cell(intent_cfg: dict, language: str, is_hard: bool, target: int,
                     already: int, seen: set, out_f, pbar, lock: asyncio.Lock):
    """Sinh cho tới khi đủ `target` mẫu duy nhất cho 1 ô (intent, language, is_hard)."""
    have = already
    max_empty_batches = 6  
    empty_streak = 0

    while have < target and empty_streak < max_empty_batches:
        remaining = target - have
        batch_size = min(BATCH_SIZE, remaining + 5) 
        candidates = await generate_batch(intent_cfg, language, is_hard, batch_size)

        added_this_round = 0
        async with lock:
            for text in candidates:
                key = normalize_for_dedup(text)
                if key in seen:
                    continue
                seen.add(key)
                record = {
                    "text": text,
                    "label": intent_cfg["label"],
                    "intent": intent_cfg["intent"],
                    "is_noise": intent_cfg["intent"] == "ignore",
                    "language": language,
                    "is_hard": is_hard,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush() 
                added_this_round += 1
                have += 1
                pbar.update(1)
                if have >= target:
                    break

        empty_streak = empty_streak + 1 if added_this_round == 0 else 0

    if have < target:
        print(f"[Cảnh báo] ({intent_cfg['intent']}, {language}, hard={is_hard}) chỉ đạt {have}/{target} "
              f"sau nhiều lần thử - model có thể đang sinh trùng lặp quá nhiều, cân nhắc tăng temperature "
              f"hoặc đa dạng hoá prompt thêm.")


async def main():
    if not API_KEY:
        print("Thiếu OPENAI_API_KEY. Set biến môi trường trước khi chạy:")
        print(r'  $env:OPENAI_API_KEY="..."   (PowerShell)')
        return

    records, seen, counts = load_existing()
    print(f"Đã có {len(records)} mẫu từ trước (nếu có) trong {OUTPUT_FILE}, sẽ resume phần còn thiếu.")


    total_needed = 0
    plan = []
    for intent_cfg in INTENTS:
        for language in LANGUAGES:
            for is_hard, per_cell_target in [(False, PER_CELL_NORMAL), (True, PER_CELL_HARD)]:
                key = (intent_cfg["intent"], language, is_hard)
                already = counts.get(key, 0)
                remaining = max(0, per_cell_target - already)
                total_needed += remaining
                plan.append((intent_cfg, language, is_hard, per_cell_target, already))

    if total_needed == 0:
        print("Đã đủ dữ liệu theo target hiện tại, không cần sinh thêm.")
        return

    print(f"Cần sinh thêm {total_needed} mẫu (target tổng: {TOTAL_TARGET}).")

    lock = asyncio.Lock()
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        with tqdm(total=total_needed, desc="Sinh dữ liệu") as pbar:
            tasks = [
                fill_cell(intent_cfg, language, is_hard, target, already, seen, out_f, pbar, lock)
                for (intent_cfg, language, is_hard, target, already) in plan
                if target - already > 0
            ]
            await asyncio.gather(*tasks)

    _, _, final_counts = load_existing()
    print("\n--- THỐNG KÊ CUỐI CÙNG ---")
    total = 0
    for intent_cfg in INTENTS:
        for language in LANGUAGES:
            for is_hard in [False, True]:
                key = (intent_cfg["intent"], language, is_hard)
                c = final_counts.get(key, 0)
                total += c
                print(f"  {intent_cfg['intent']:<12} {language:<3} hard={is_hard!s:<5} : {c}")
    print(f"TỔNG: {total} mẫu -> {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())