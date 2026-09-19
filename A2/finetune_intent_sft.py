# -*- coding: utf-8 -*-
"""
C2.1 — Fine-tune SLM Intent Extraction bằng SFT (LoRA), giống pipeline đã
dùng cho Router A1.3 (finetune_router_sft.py), áp dụng cho tác vụ tách
{subject}/{action}/{context}.

Base model: Qwen2.5-0.5B-Instruct (< 3B, đúng yêu cầu đề bài).

Cài đặt cần thêm (nếu chưa có từ lúc làm A1):
    pip install peft accelerate datasets

Chạy:
    python3 finetune_intent_sft.py --epochs 3 --batch-size 8
Input: c2_data/intent_dataset.jsonl (sinh bởi generate_intent_dataset.py)
Output: ./intent_sft_lora/ (adapter), ./intent_sft_merged/ (bản đã merge)
        c2_data/intent_test_set.jsonl (test set giữ lại để evaluate_intent_extractor.py dùng)
"""
from __future__ import annotations
import argparse
import json
import os
import random

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
HERE = os.path.dirname(__file__)
DATA_FILE = os.path.join(HERE, "c2_data", "intent_dataset.jsonl")
TEST_OUT = os.path.join(HERE, "c2_data", "intent_test_set.jsonl")
LORA_OUT = os.path.join(HERE, "intent_sft_lora")
MERGED_OUT = os.path.join(HERE, "intent_sft_merged")

SYSTEM_PROMPT = """Bạn là bộ phân tích câu nói của khách hàng quán cà phê Highlands.
Tách câu nói thành đúng 3 phần, trả về DUY NHẤT 1 JSON object với 3 khoá:
"subject", "action", "context". "action" phải giữ nguyên nội dung câu hỏi/yêu
cầu gốc, chỉ bỏ từ đệm lịch sự. "context" để rỗng "" nếu không có yếu tố thời
gian/số lượng/dịp đặc biệt."""


def load_and_split(test_ratio=0.15, seed=42):
    rows = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    random.Random(seed).shuffle(rows)
    n_test = max(1, int(len(rows) * test_ratio))
    test_rows = rows[:n_test]
    train_rows = rows[n_test:]
    return train_rows, test_rows


def build_example(tokenizer, row):
    target = json.dumps(
        {"subject": row["subject"], "action": row["action"], "context": row["context"]},
        ensure_ascii=False,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["text"]},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + target + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    labels = labels[: len(full_ids)]
    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


def main(epochs: int, batch_size: int, lr: float):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows, test_rows = load_and_split()
    print(f"Train: {len(train_rows)} samples | Test (giữ lại để evaluate): {len(test_rows)} samples")

    os.makedirs(os.path.dirname(TEST_OUT), exist_ok=True)
    with open(TEST_OUT, "w", encoding="utf-8") as f:
        for r in test_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_examples = [build_example(tokenizer, r) for r in train_rows]
    train_ds = Dataset.from_list(train_examples)

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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda"
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=os.path.join(HERE, "intent_sft_ckpt"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collate)
    trainer.train()

    model.save_pretrained(LORA_OUT)
    tokenizer.save_pretrained(LORA_OUT)
    print(f"Đã lưu LoRA adapter tại {LORA_OUT}")

    print("Đang merge LoRA vào base model...")
    merged = model.merge_and_unload()
    merged.save_pretrained(MERGED_OUT)
    tokenizer.save_pretrained(MERGED_OUT)
    print(f"Đã lưu model đã merge tại {MERGED_OUT}")
    print("\n*** QUAN TRỌNG: chạy verify_merged_checkpoint.py NGAY để kiểm tra "
          "checkpoint có bị hỏng không (bài học từ sự cố Router A1) trước khi "
          "dùng merged model này ***")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.lr)