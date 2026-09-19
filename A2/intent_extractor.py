# -*- coding: utf-8 -*-
"""
Phần C2.1 — SLM Intent Extraction (yêu cầu <3B, đề bài trang 12).

GHI CHÚ CẬP NHẬT: Đề bài yêu cầu "bắt buộc fine-tune" một SLM cho tác vụ
này. Bản đầu tiên dùng PROMPTING với model base (chưa fine-tune) và có lỗi
hallucination thật (xem báo cáo Phần C, mục 3). Đã khắc phục bằng fine-tune
LoRA SFT thật (xem finetune_intent_sft.py, generate_intent_dataset.py) —
checkpoint đã merge tại ./intent_sft_merged, đã xác nhận KHÔNG hỏng
(verify_merged_checkpoint.py) và đo accuracy 98.6% trên test set giữ riêng
(evaluate_intent_extractor.py) — vượt ngưỡng đề bài (>=90%). MODEL_PATH bên
dưới nay trỏ vào checkpoint đã fine-tune, không còn dùng model base.

Tách câu nói khách hàng thành 3 phần theo đúng cấu trúc đề bài:
  - subject   : {Chủ ngữ} — ai đang nói / yêu cầu
  - action    : {Hành động / Yêu cầu} — nội dung cốt lõi, ĐÃ loại bỏ từ đệm
                lịch sự/chào hỏi. Đây là phần dùng làm CACHE KEY.
  - context   : {Ngữ cảnh / Thời gian / Bối cảnh} — yếu tố phụ trợ nếu có.

Đặt vào /mnt/d/HEVO/A2 (cùng thư mục router_agent_fast.py) để dùng chung
GPU/model đã nạp.
"""
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "intent_sft_merged")
BASE_MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"  # fallback nếu chưa fine-tune ở máy này

if not os.path.isdir(MODEL_PATH):
    MODEL_PATH = BASE_MODEL_PATH

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="cuda",
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )
        _model.eval()
    return _tokenizer, _model


SYSTEM_PROMPT = """Bạn là bộ phân tích câu nói của khách hàng quán cà phê Highlands.
Tách câu nói thành đúng 3 phần, trả về DUY NHẤT 1 JSON object với 3 khoá:
- "subject": ai đang nói / yêu cầu (vd "anh", "em", "chị"; mặc định "khách" nếu không rõ).
- "action": phần yêu cầu/câu hỏi CỐT LÕI, đã loại bỏ hết các từ đệm lịch sự,
  chào hỏi, cảm ơn (như "cho em hỏi", "làm ơn", "excuse me", "cảm ơn shop",
  "vậy ạ", "hỏi giúp bạn mình", "mình muốn biết", "không biết ... không").
  PHẢI giữ nguyên nội dung/ý nghĩa câu hỏi gốc, chỉ bỏ từ đệm, vì phần này
  dùng làm khoá tra cứu cache.
- "context": yếu tố phụ trợ như thời gian, số lượng người, dịp đặc biệt;
  chuỗi rỗng "" nếu câu không có yếu tố này.
Chỉ trả về JSON, không thêm chữ nào khác.

Ví dụ 1:
Input: "Cho anh đặt bàn tiệc sinh nhật vào ngày mai lúc 7h em nhé"
Output: {"subject": "anh", "action": "đặt bàn tiệc sinh nhật", "context": "ngày mai lúc 7h"}

Ví dụ 2:
Input: "Cho em hỏi wifi quán tên gì, mật khẩu là gì?"
Output: {"subject": "em", "action": "wifi quán tên gì, mật khẩu là gì", "context": ""}

Ví dụ 3:
Input: "Làm ơn cho hỏi quán mấy giờ đóng cửa vậy ạ?"
Output: {"subject": "khách", "action": "quán mấy giờ đóng cửa", "context": ""}
"""


def extract_intent(user_query: str, max_new_tokens: int = 96):
    """Trả về (parsed_dict, latency_ms, raw_generated_text)."""
    tokenizer, model = _load()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - t0) * 1000

    gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    match = re.search(r"\{.*\}", gen_text, re.DOTALL)
    if not match:
        return {"subject": "khách", "action": user_query.strip(), "context": ""}, latency_ms, gen_text
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"subject": "khách", "action": user_query.strip(), "context": ""}, latency_ms, gen_text

    parsed.setdefault("subject", "khách")
    parsed.setdefault("action", user_query.strip())
    parsed.setdefault("context", "")
    return parsed, latency_ms, gen_text


if __name__ == "__main__":
    print("Đang nạp model...")
    _load()
    print("Sẵn sàng. Test nhanh vài câu:\n")

    TEST_QUERIES = [
        "Cho em hỏi wifi quán tên gì, mật khẩu là gì?",
        "Anh/chị ơi, quán mở cửa mấy giờ, đóng cửa mấy giờ?",
        "Cho anh 1 ly cà phê sữa đá size L mang đi nhé.",
        "Cho anh đặt bàn tiệc sinh nhật vào ngày mai lúc 7h em nhé",
        "Làm ơn cho hỏi có được mang thú cưng vào quán không?",
    ]
    for q in TEST_QUERIES:
        parsed, latency_ms, raw = extract_intent(q)
        print(f"IN : {q}")
        print(f"OUT: {json.dumps(parsed, ensure_ascii=False)}  ({latency_ms:.1f}ms)")
        print()