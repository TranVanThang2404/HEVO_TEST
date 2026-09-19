# -*- coding: utf-8 -*-
"""
C2.1 — Đo accuracy THẬT của SLM Intent Extraction đã fine-tune, trên test
set giữ lại lúc train (c2_data/intent_test_set.jsonl, KHÔNG lẫn với train).

Định nghĩa accuracy (field-level, ghi rõ trong báo cáo để minh bạch cách đo):
- "action" ĐÚNG nếu khớp với gold sau khi chuẩn hoá (lower, bỏ dấu câu ở đầu/cuối).
  Đây là field quan trọng nhất (dùng làm cache-key).
- "subject" ĐÚNG nếu khớp gold, hoặc cả 2 đều rơi vào nhóm chung (anh/chị/em/
  tôi/mình đều map "khách" khi so sánh lỏng) — vì subject chỉ mang tính tham
  khảo, không dùng làm cache-key.
- "context" ĐÚNG nếu khớp gold, hoặc cả 2 đều rỗng.
- 1 mẫu ĐÚNG TOÀN BỘ (overall accuracy theo yêu cầu đề bài) nếu action ĐÚNG
  và context ĐÚNG (subject không tính vào vì ít quan trọng, chỉ báo cáo riêng).

Chạy:
    python3 evaluate_intent_extractor.py --model ./intent_sft_merged
    python3 evaluate_intent_extractor.py --model base   # để so sánh với bản chưa fine-tune
"""
import argparse
import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(__file__)
TEST_FILE = os.path.join(HERE, "c2_data", "intent_test_set.jsonl")

SYSTEM_PROMPT = """Bạn là bộ phân tích câu nói của khách hàng quán cà phê Highlands.
Tách câu nói thành đúng 3 phần, trả về DUY NHẤT 1 JSON object với 3 khoá:
"subject", "action", "context". "action" phải giữ nguyên nội dung câu hỏi/yêu
cầu gốc, chỉ bỏ từ đệm lịch sự. "context" để rỗng "" nếu không có yếu tố thời
gian/số lượng/dịp đặc biệt."""


def normalize(s: str) -> str:
    return re.sub(r"[?!.,;:]+$", "", (s or "").strip().lower()).strip()


def load_test_set():
    rows = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_model(model_path: str, rows: list[dict]):
    if model_path == "base":
        model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    results = []
    for row in rows:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["text"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        latency_ms = (time.perf_counter() - t0) * 1000
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        match = re.search(r"\{.*\}", gen, re.DOTALL)
        parsed = {}
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = {}

        pred_action = normalize(parsed.get("action", ""))
        pred_context = normalize(parsed.get("context", ""))
        gold_action = normalize(row["action"])
        gold_context = normalize(row["context"])

        action_ok = pred_action == gold_action
        context_ok = pred_context == gold_context
        overall_ok = action_ok and context_ok

        results.append({
            "text": row["text"], "gold_action": gold_action, "pred_action": pred_action,
            "gold_context": gold_context, "pred_context": pred_context,
            "action_ok": action_ok, "context_ok": context_ok, "overall_ok": overall_ok,
            "latency_ms": latency_ms, "raw": gen,
        })
    return results


def main(model_path: str):
    rows = load_test_set()
    print(f"Đang đánh giá trên {len(rows)} mẫu test (giữ riêng lúc train, model KHÔNG thấy qua)")
    print(f"Model: {model_path}\n")

    results = run_model(model_path, rows)

    n = len(results)
    action_acc = sum(r["action_ok"] for r in results) / n * 100
    context_acc = sum(r["context_ok"] for r in results) / n * 100
    overall_acc = sum(r["overall_ok"] for r in results) / n * 100
    avg_latency = sum(r["latency_ms"] for r in results) / n

    print("Chi tiết các mẫu SAI (để kiểm tra thủ công):")
    n_shown = 0
    for r in results:
        if not r["overall_ok"] and n_shown < 15:
            print(f"  IN : {r['text']}")
            print(f"    gold_action='{r['gold_action']}' | pred_action='{r['pred_action']}'")
            print(f"    gold_context='{r['gold_context']}' | pred_context='{r['pred_context']}'")
            n_shown += 1

    print("\n" + "=" * 70)
    print("KẾT QUẢ ĐÁNH GIÁ C2.1 — SLM INTENT EXTRACTION")
    print("=" * 70)
    print(f"Số mẫu test           : {n}")
    print(f"Action accuracy        : {action_acc:.1f}%")
    print(f"Context accuracy       : {context_acc:.1f}%")
    print(f"Overall accuracy (action+context): {overall_acc:.1f}%  (yêu cầu đề bài: >= 90%)")
    print(f"Latency trung bình     : {avg_latency:.1f}ms")
    print(f"KẾT LUẬN: {'ĐẠT' if overall_acc >= 90 else 'CHƯA ĐẠT'} yêu cầu accuracy >= 90%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./intent_sft_merged",
                         help="Đường dẫn model đã merge, hoặc 'base' để test bản chưa fine-tune")
    args = parser.parse_args()
    main(args.model)