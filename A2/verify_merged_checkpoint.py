# -*- coding: utf-8 -*-
"""
Kiểm tra checkpoint đã merge có bị HỎNG hay không, TRƯỚC KHI dùng nó cho bất
kỳ việc gì khác. Bài học từ sự cố Router A1 (router_finetuned_output/router_merged
bị hỏng, sinh token rác kể cả với văn bản thường — vocab/embedding size vẫn
khớp bình thường nên KHÔNG thể chỉ kiểm tra shape, phải generate thử).

Chạy NGAY sau khi finetune_intent_sft.py xong:
    python3 verify_merged_checkpoint.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./intent_sft_merged"

print(f"Đang nạp {MODEL_PATH} để kiểm tra...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="cuda")
model.eval()

print(f"tokenizer vocab len : {len(tokenizer)}")
print(f"model embedding shape: {model.get_input_embeddings().weight.shape}")
print(f"config.vocab_size    : {model.config.vocab_size}")

# --- Test 1: văn bản THƯỜNG, KHÔNG liên quan gì tới task intent extraction ---
# Đây là bài test quan trọng nhất — nếu checkpoint hỏng, kể cả câu đơn giản
# này cũng sẽ ra token rác.
plain_prompt = "Xin chào, hôm nay trời đẹp quá"
inputs = tokenizer(plain_prompt, return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
plain_continuation = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\n[Test 1 - văn bản thường]")
print(f"Input : {plain_prompt}")
print(f"Output: {plain_continuation}")

is_garbage = any(c in plain_continuation for c in "中国日本한국") or len(set(plain_continuation.strip())) <= 2
if is_garbage:
    print("\n*** CẢNH BÁO: OUTPUT NGHI NGỜ LÀ RÁC (lặp ký tự lạ) — CHECKPOINT CÓ THỂ ĐÃ HỎNG ***")
    print("*** KHÔNG dùng checkpoint này cho intent_extractor.py — quay lại dùng base model ***")
else:
    print("\n[OK] Output là văn bản tiếng Việt bình thường, có vẻ checkpoint KHÔNG bị hỏng.")

# --- Test 2: câu intent extraction thật, kiểm tra định dạng JSON đúng ---
print("\n[Test 2 - câu intent extraction thật]")
SYSTEM_PROMPT = """Bạn là bộ phân tích câu nói của khách hàng quán cà phê Highlands.
Tách câu nói thành đúng 3 phần, trả về DUY NHẤT 1 JSON object với 3 khoá:
"subject", "action", "context". "action" phải giữ nguyên nội dung câu hỏi/yêu
cầu gốc, chỉ bỏ từ đệm lịch sự. "context" để rỗng "" nếu không có yếu tố thời
gian/số lượng/dịp đặc biệt."""

TEST_QUERIES = [
    "Cho em hỏi wifi quán tên gì, mật khẩu là gì?",
    "Anh/chị ơi, quán mở cửa mấy giờ, đóng cửa mấy giờ?",
    "Cho anh 1 ly cà phê sữa đá size L mang đi nhé.",
    "Cho anh đặt bàn tiệc sinh nhật vào ngày mai lúc 7h em nhé",
]
for q in TEST_QUERIES:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"IN : {q}")
    print(f"OUT: {gen}\n")

print("=" * 70)
print("Nếu Test 1 KHÔNG cảnh báo và Test 2 ra JSON hợp lệ, đúng cấu trúc")
print("-> checkpoint OK, có thể dùng cho evaluate_intent_extractor.py")
print("=" * 70)