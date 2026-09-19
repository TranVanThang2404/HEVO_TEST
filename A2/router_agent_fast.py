import sys
sys.stdout.reconfigure(encoding='utf-8')

import time
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"

print("Đang nạp mô hình lên card RTX 3060...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cuda",
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
)
model.eval()

SYSTEM_PROMPT = """Bạn là Router Agent của Highlands Coffee. Khách có thể nói tiếng Việt hoặc tiếng Anh.
Phân loại ý định của khách hàng thành 1 trong 4 loại, bất kể ngôn ngữ khách dùng:
- "order": Đặt hàng, tính tiền, gọi thêm món. Ví dụ: "Cho 1 ly bạc xỉu size M", "Tính tiền giúp em", "I'd like a large iced latte".
- "consultant": Nhờ tư vấn, gợi ý món. Ví dụ: "Có món nào ngọt mà không quá béo không?", "What do you recommend for a hot day?".
- "faq": Hỏi thông tin CỦA QUÁN (wifi, giờ mở cửa, địa chỉ, chỗ gửi xe...). Chỉ chọn faq khi câu hỏi thật sự hỏi về quán. Ví dụ: "Wifi pass gì vậy em?", "What time do you close?".
- "ignore": Câu nói KHÔNG mang yêu cầu/câu hỏi nào tới quán — chào hỏi vu vơ, test mic/kết nối, tiếng ồn, cười đùa, gọi tên suông. Đây KHÔNG phải faq dù có dạng câu hỏi. Ví dụ: "Alo alo test mic", "haha".
Trả lời đúng 1 nhãn duy nhất trong 4 nhãn trên."""

LABELS = ["order", "consultant", "faq", "ignore"]


def _common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


_NL_IDS = tokenizer.encode("\n", add_special_tokens=False)
LABEL_IDS = []
for _label in LABELS:
    _combo_ids = tokenizer.encode("\n" + _label, add_special_tokens=False)
    _common = _common_prefix_len(_NL_IDS, _combo_ids)
    LABEL_IDS.append(_combo_ids[_common:])

print("Token nhãn đã xác định:", {lbl: ids for lbl, ids in zip(LABELS, LABEL_IDS)})


FIXED_LEN = 340


def build_prompt_ids(user_query: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@torch.inference_mode()
def _raw_scores(user_query: str):
    prompt_ids = build_prompt_ids(user_query)
    sequences = [prompt_ids + lbl_ids for lbl_ids in LABEL_IDS]

    seq_max = max(len(s) for s in sequences)
    pad_len = max(FIXED_LEN, seq_max)
    if pad_len > FIXED_LEN:
        print(f"[Cảnh báo] câu dài hơn FIXED_LEN ({pad_len} > {FIXED_LEN}), nên tăng FIXED_LEN để giữ tốc độ ổn định.")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    input_ids = torch.full((len(LABELS), pad_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((len(LABELS), pad_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        input_ids[i, :len(seq)] = torch.tensor(seq)
        attn_mask[i, :len(seq)] = 1
    input_ids = input_ids.to(model.device)
    attn_mask = attn_mask.to(model.device)

    start_time = time.perf_counter()
    logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
    log_probs = F.log_softmax(logits.float(), dim=-1)

    p_len = len(prompt_ids)
    scores = []
    for i, lbl_ids in enumerate(LABEL_IDS):
        total_lp = 0.0
        for j, tok_id in enumerate(lbl_ids):
            pos = p_len + j - 1
            total_lp += log_probs[i, pos, tok_id].item()
        scores.append(total_lp / len(lbl_ids))
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000
    return scores, latency_ms

_CALIBRATION_QUERIES = ["N/A", "xyz123", "。。。", " "]
print("Đang tính calibration baseline...")
_baseline_accum = [0.0] * len(LABELS)
for _cq in _CALIBRATION_QUERIES:
    _s, _ = _raw_scores(_cq)
    for _i in range(len(LABELS)):
        _baseline_accum[_i] += _s[_i]
BASELINE_SCORES = [v / len(_CALIBRATION_QUERIES) for v in _baseline_accum]
print("Baseline theo nhãn:", dict(zip(LABELS, [round(b, 3) for b in BASELINE_SCORES])))


def classify_intent(user_query: str):
    scores, latency_ms = _raw_scores(user_query)
    calibrated = [s - b for s, b in zip(scores, BASELINE_SCORES)]
    best_idx = max(range(len(LABELS)), key=lambda i: calibrated[i])
    return {"action": LABELS[best_idx]}, latency_ms


if __name__ == "__main__":
    # Danh sách ứng viên: nhiều câu cho mỗi nhãn, để tìm ra bộ 4 câu mà
    # router 0.5B base này PHÂN LOẠI ĐÚNG cả 4 loại, dùng cho demo video.
    CANDIDATES = {
        "order": [
            "Cho anh 1 ly cà phê sữa đá size L mang đi nhé.",
            "Cho em 2 ly trà đào size M, 1 mang đi 1 uống tại chỗ.",
            "Tính tiền giúp em với.",
            "Em đặt 1 ly bạc xỉu đá, ít đường nha.",
            "I'd like to order a large iced latte, please.",
        ],
        "consultant": [
            "Tư vấn giúp em món gì mát mát cho mùa hè với.",
            "Có món nào ngọt mà không quá béo không em?",
            "What do you recommend for someone who doesn't like bitter coffee?",
            "Quán có món gì đang hot không tư vấn giúp anh?",
        ],
        "faq": [
            "Quán mình wifi pass là gì vậy em?",
            "Quán mấy giờ đóng cửa vậy em?",
            "Ở đây có chỗ gửi xe không em?",
            "What time do you open tomorrow?",
        ],
        "ignore": [
            "Alo alo, nghe rõ không haha.",
            "Test mic thử coi.",
            "haha đùa thôi.",
            "ừm... để coi...",
        ],
    }

    print("Đang warm-up GPU...")
    for _ in range(3):
        classify_intent("warm up")

    print("\n--- TÌM CÂU DEMO PHÙ HỢP (router 0.5B base) ---")
    for true_label, queries in CANDIDATES.items():
        print(f"\n=== Nhãn thật: {true_label} ===")
        for q in queries:
            result, latency = classify_intent(q)
            mark = "✅" if result["action"] == true_label else "❌"
            print(f"{mark} predicted={result['action']:<11} ({latency:5.1f}ms)  '{q}'")