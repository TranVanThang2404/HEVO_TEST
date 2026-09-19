"""
A1.3 - Fine-tuning Router Agent (SFT + LoRA) trên GPU cloud (Modal, H100 Trial)

Cách chạy:
    pip install modal
    modal setup                      # đăng nhập lần đầu (mở browser)
    modal run finetune_router.py --data-file router_dataset.jsonl

Modal sẽ tự động:
  1. Dựng 1 container sạch với đúng thư viện cần thiết (không cần cài tay như WSL).
  2. Upload router_dataset.jsonl từ máy bạn lên container.
  3. Cấp 1 GPU H100 chạy fine-tune.
  4. Lưu kết quả (model đã merge LoRA + báo cáo benchmark) vào Modal Volume,
     rồi tải ngược về máy bạn vào thư mục ./router_finetuned_output/.

Thiết kế:
  - Base model: Qwen2.5-0.5B-Instruct (đúng model đang serving ở A1.1).
  - SFT theo đúng format inference thực tế: system prompt + user query -> assistant
    trả về ĐÚNG 1 nhãn (order/consultant/faq/ignore), giống hệt cách router_agent_fast.py
    đang chấm điểm logit -> đảm bảo việc fine-tune cải thiện trực tiếp độ chính xác
    của cơ chế đang dùng trong production, không phải một cách train khác biệt.
  - Chỉ tính loss trên phần token nhãn (assistant), che (-100) phần prompt.
  - LoRA (không train full weight) -> nhẹ, nhanh, đủ cho model 0.5B.
  - Chia train/val/test theo tỷ lệ 80/10/10, GIỮ NGUYÊN tỷ lệ hard/normal và
    ngôn ngữ ở mỗi tập (stratified) để test set vẫn cân bằng đúng yêu cầu đề bài.
  - Benchmark đầy đủ: accuracy tổng, accuracy riêng hard samples, confusion matrix,
    per-class precision/recall/F1, và vài ví dụ lỗi cụ thể để phân tích.
"""

import json
import random
from collections import defaultdict

import modal

# ==================== MODAL SETUP ====================

app = modal.App("router-finetune")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "peft==0.12.0",
        "accelerate==0.33.0",
        "datasets==2.20.0",
        "scikit-learn==1.5.1",
        "matplotlib==3.9.1",
        "numpy<2.0",
    )
)

volume = modal.Volume.from_name("router-finetune-vol", create_if_missing=True)
VOL_PATH = "/vol"

MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
# v3: sau 2 lần thử với Qwen2.5-0.5B (chỉnh LoRA rank/epoch/oversample) vẫn loanh quanh
# 88-89% acc, 67-70% hard acc -> nghi ngờ 0.5B không đủ capacity phân biệt 4 lớp ngữ nghĩa
# gần nhau. Đổi sang 1.5B (đề bài A1.1 cho phép 0.5B-1.5B) để kiểm chứng.
# LƯU Ý: nếu bản 1.5B đạt mục tiêu, cần cập nhật lại router_agent_fast.py (A1.1) dùng
# đúng model 1.5B này để training/serving nhất quán, và benchmark lại latency (mục tiêu
# vẫn <=200ms/<=100ms, model to hơn 3x nên cần đo lại thực tế trên RTX 3060).
LABELS = ["order", "consultant", "faq", "ignore"]

# Giữ NGUYÊN VĂN system prompt đang dùng ở production (router_agent_fast.py) để
# việc fine-tune khớp đúng với cách model sẽ được gọi lúc inference thật.
SYSTEM_PROMPT = """Bạn là Router Agent của Highlands Coffee. Khách có thể nói tiếng Việt hoặc tiếng Anh.
Phân loại ý định của khách hàng thành 1 trong 4 loại, bất kể ngôn ngữ khách dùng:
- "order": Đặt hàng, tính tiền, gọi thêm món. Ví dụ: "Cho 1 ly bạc xỉu size M", "Tính tiền giúp em", "I'd like a large iced latte".
- "consultant": Nhờ tư vấn, gợi ý món. Ví dụ: "Có món nào ngọt mà không quá béo không?", "What do you recommend for a hot day?".
- "faq": Hỏi thông tin CỦA QUÁN (wifi, giờ mở cửa, địa chỉ, chỗ gửi xe...). Chỉ chọn faq khi câu hỏi thật sự hỏi về quán. Ví dụ: "Wifi pass gì vậy em?", "What time do you close?".
- "ignore": Câu nói KHÔNG mang yêu cầu/câu hỏi nào tới quán — chào hỏi vu vơ, test mic/kết nối, tiếng ồn, cười đùa, gọi tên suông. Đây KHÔNG phải faq dù có dạng câu hỏi. Ví dụ: "Alo alo test mic", "haha".
Trả lời đúng 1 nhãn duy nhất trong 4 nhãn trên."""


def stratified_split(records, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Chia train/val/test, giữ đúng tỷ lệ intent + language + is_hard ở mỗi tập."""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for r in records:
        key = (r["intent"], r["language"], r["is_hard"])
        buckets[key].append(r)

    train, val, test = [], [], []
    for key, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ==================== TRAINING (chạy trong container Modal) ====================

@app.function(
    image=image,
    gpu="H100",
    volumes={VOL_PATH: volume},
    timeout=3600,
)
def train_and_evaluate(dataset_bytes: bytes):
    import os
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    # ---- Load & split dữ liệu ----
    records = [json.loads(line) for line in dataset_bytes.decode("utf-8").splitlines() if line.strip()]
    train_records, val_records, test_records = stratified_split(records)
    print(f"Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def build_example(rec):
        """Tokenize prompt+label, mask phần prompt (-100) để chỉ tính loss trên nhãn."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rec["text"]},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_messages = messages + [{"role": "assistant", "content": rec["intent"]}]
        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    # v3: BỎ oversample hard x3 (thử ở v2) — kết quả cho thấy nó làm model bị lệch,
    # đoán tràn lan sang "consultant" (precision consultant rớt còn 0.79 dù recall lên 0.93).
    # Quay lại train bình thường trên đúng phân bố gốc (đã làm sạch nhãn hard qua
    # clean_router_dataset.py).
    print(f"Train (không oversample): {len(train_records)}")

    train_ds = Dataset.from_list([build_example(r) for r in train_records])
    val_ds = Dataset.from_list([build_example(r) for r in val_records])

    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        pad_id = tokenizer.pad_token_id
        input_ids, attn, labels = [], [], []
        for x in batch:
            pad_n = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad_n)
            attn.append(x["attention_mask"] + [0] * pad_n)
            labels.append(x["labels"] + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn),
            "labels": torch.tensor(labels),
        }

    # ---- Load model + gắn LoRA ----
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda")
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=f"{VOL_PATH}/checkpoints",
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
    )
    trainer.train()

    # ---- Merge LoRA vào base model để dễ quantize/serving sau này ----
    merged_model = model.merge_and_unload()
    merged_dir = f"{VOL_PATH}/router_merged"
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    volume.commit()

    # ==================== EVALUATE trên test set (giữ nguyên) ====================
    merged_model.eval()

    def _common_prefix_len(a, b):
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

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
        pad_id = tokenizer.pad_token_id
        input_ids = torch.full((len(LABELS), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(LABELS), max_len), dtype=torch.long)
        for i, seq in enumerate(sequences):
            input_ids[i, :len(seq)] = torch.tensor(seq)
            attn[i, :len(seq)] = 1
        input_ids, attn = input_ids.to(merged_model.device), attn.to(merged_model.device)

        logits = merged_model(input_ids=input_ids, attention_mask=attn).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        p_len = len(prompt_ids)
        scores = []
        for i, lbl in enumerate(label_ids):
            total = 0.0
            for j, tok in enumerate(lbl):
                total += log_probs[i, p_len + j - 1, tok].item()
            scores.append(total / len(lbl))
        return LABELS[scores.index(max(scores))]

    y_true, y_pred, is_hard_flags, mistakes = [], [], [], []
    for rec in test_records:
        pred = classify(rec["text"])
        y_true.append(rec["intent"])
        y_pred.append(pred)
        is_hard_flags.append(rec["is_hard"])
        if pred != rec["intent"]:
            mistakes.append({"text": rec["text"], "true": rec["intent"], "pred": pred,
                              "is_hard": rec["is_hard"], "language": rec["language"]})

    overall_acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    hard_idx = [i for i, h in enumerate(is_hard_flags) if h]
    normal_idx = [i for i, h in enumerate(is_hard_flags) if not h]
    hard_acc = sum(y_true[i] == y_pred[i] for i in hard_idx) / max(len(hard_idx), 1)
    normal_acc = sum(y_true[i] == y_pred[i] for i in normal_idx) / max(len(normal_idx), 1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    report = {
        "test_set_size": len(test_records),
        "overall_accuracy": overall_acc,
        "hard_sample_accuracy": hard_acc,
        "normal_sample_accuracy": normal_acc,
        "per_class": {
            lbl: {"precision": float(precision[i]), "recall": float(recall[i]),
                  "f1": float(f1[i]), "support": int(support[i])}
            for i, lbl in enumerate(LABELS)
        },
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        "sample_mistakes": mistakes[:30],
        "target_accuracy_92pct_met": overall_acc >= 0.92,
        "target_hard_75pct_met": hard_acc >= 0.75,
    }

    with open(f"{VOL_PATH}/eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Vẽ confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=45)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (Acc={overall_acc:.3f}, Hard Acc={hard_acc:.3f})")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(f"{VOL_PATH}/confusion_matrix.png", dpi=150)

    volume.commit()
    print(json.dumps({k: v for k, v in report.items() if k != "sample_mistakes"}, ensure_ascii=False, indent=2))
    return report


@app.local_entrypoint()
def main(data_file: str = "router_dataset.jsonl"):
    with open(data_file, "rb") as f:
        data_bytes = f.read()

    print(f"Đang gửi {data_file} lên Modal, khởi động GPU H100...")
    report = train_and_evaluate.remote(data_bytes)

    print("\n=== KẾT QUẢ CUỐI CÙNG ===")
    print(f"Overall accuracy : {report['overall_accuracy']*100:.2f}% "
          f"({'ĐẠT' if report['target_accuracy_92pct_met'] else 'CHƯA ĐẠT'} mục tiêu ≥92%)")
    print(f"Hard-sample acc  : {report['hard_sample_accuracy']*100:.2f}% "
          f"({'ĐẠT' if report['target_hard_75pct_met'] else 'CHƯA ĐẠT'} mục tiêu ≥75%)")
    print("\nPer-class F1:")
    for lbl, m in report["per_class"].items():
        print(f"  {lbl:<12} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")

    print("\nĐang tải model đã fine-tune + báo cáo về máy (thư mục ./router_finetuned_output/)...")
    import subprocess
    subprocess.run(["modal", "volume", "get", "router-finetune-vol", "router_merged",
                     "router_finetuned_output/router_merged", "--force"], check=False)
    subprocess.run(["modal", "volume", "get", "router-finetune-vol", "eval_report.json",
                     "router_finetuned_output/eval_report.json", "--force"], check=False)
    subprocess.run(["modal", "volume", "get", "router-finetune-vol", "confusion_matrix.png",
                     "router_finetuned_output/confusion_matrix.png", "--force"], check=False)
    print("Xong. Xem router_finetuned_output/eval_report.json và confusion_matrix.png")
