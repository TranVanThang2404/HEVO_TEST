"""
Biến thể của test_router_logit_scoring.py nhưng test TRỰC TIẾP model FP16
router_merged (CHƯA quantize AWQ) — để cô lập xem lỗi (bias nặng về "faq",
33% accuracy) đến từ bước AWQ hay đã có sẵn từ bước fine-tune/merge.

Chạy: python3 test_router_logit_scoring_fp16.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "router_finetuned_output/router_merged/router_merged"
LABELS = ["order", "consultant", "faq", "ignore"]

SYSTEM_PROMPT = """Bạn là Router Agent của Highlands Coffee. Khách có thể nói tiếng Việt hoặc tiếng Anh.
Phân loại ý định của khách hàng thành 1 trong 4 loại, bất kể ngôn ngữ khách dùng:
- "order": Đặt hàng, tính tiền, gọi thêm món. Ví dụ: "Cho 1 ly bạc xỉu size M", "Tính tiền giúp em", "I'd like a large iced latte".
- "consultant": Nhờ tư vấn, gợi ý món. Ví dụ: "Có món nào ngọt mà không quá béo không?", "What do you recommend for a hot day?".
- "faq": Hỏi thông tin CỦA QUÁN (wifi, giờ mở cửa, địa chỉ, chỗ gửi xe...). Chỉ chọn faq khi câu hỏi thật sự hỏi về quán. Ví dụ: "Wifi pass gì vậy em?", "What time do you close?".
- "ignore": Câu nói KHÔNG mang yêu cầu/câu hỏi nào tới quán — chào hỏi vu vơ, test mic/kết nối, tiếng ồn, cười đùa, gọi tên suông. Đây KHÔNG phải faq dù có dạng câu hỏi. Ví dụ: "Alo alo test mic", "haha".
Trả lời đúng 1 nhãn duy nhất trong 4 nhãn trên."""

TEST_CASES = [
    ("Cho anh 1 ly cà phê sữa đá size L mang đi nhé.", "order"),
    ("Có món nào ngọt mà không quá béo không?", "consultant"),
    ("Wifi pass là gì vậy em?", "faq"),
    ("Alo alo test mic", "ignore"),
    ("Quán mở cửa mấy giờ vậy?", "faq"),
    ("Cho tôi 2 ly trà sữa trân châu ship tới 123 Lê Lợi.", "order"),
]


def _common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main():
    print(f"[Test FP16] Đang nạp {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to("cuda")
    model.eval()
    device = "cuda"

    nl_ids = tokenizer.encode("\n", add_special_tokens=False)
    label_ids = []
    for lbl in LABELS:
        combo = tokenizer.encode("\n" + lbl, add_special_tokens=False)
        common = _common_prefix_len(nl_ids, combo)
        label_ids.append(combo[common:])

    @torch.inference_mode()
    def classify(text):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        sequences = [prompt_ids + lbl for lbl in label_ids]
        max_len = max(len(s) for s in sequences)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        input_ids = torch.full((len(LABELS), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(LABELS), max_len), dtype=torch.long)
        for i, seq in enumerate(sequences):
            input_ids[i, :len(seq)] = torch.tensor(seq)
            attn[i, :len(seq)] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        p_len = len(prompt_ids)
        scores = []
        for i, lbl in enumerate(label_ids):
            total = 0.0
            for j, tok in enumerate(lbl):
                total += log_probs[i, p_len + j - 1, tok].item()
            scores.append(total / len(lbl))
        best_idx = scores.index(max(scores))
        return LABELS[best_idx], dict(zip(LABELS, scores))

    print("\n" + "=" * 70)
    correct = 0
    for text, expected in TEST_CASES:
        pred, scores = classify(text)
        status = "OK" if pred == expected else "SAI"
        if pred == expected:
            correct += 1
        score_str = ", ".join(f"{k}={v:.2f}" for k, v in scores.items())
        print(f"[{status}] '{text}'")
        print(f"       Dự đoán: {pred} (đúng: {expected}) | scores: {score_str}")

    print("=" * 70)
    print(f"Kết quả FP16 (chưa quantize): {correct}/{len(TEST_CASES)} đúng ({correct/len(TEST_CASES)*100:.0f}%)")
    print("\nSo sánh với kết quả AWQ (33%, thiên vị 'faq'):")
    print("- Nếu FP16 CŨNG thấp/thiên vị tương tự -> lỗi từ fine-tune/merge, KHÔNG phải AWQ.")
    print("- Nếu FP16 cao (đúng như báo cáo Modal ~90%+) -> lỗi PHÁT SINH ở bước quantize AWQ,")
    print("  dù đã sửa version torch/transformers, vẫn còn nguyên nhân khác cần tìm.")


if __name__ == "__main__":
    main()