# -*- coding: utf-8 -*-
"""
C2.1 — Data Generation cho SLM Intent Extraction (fine-tune thật).

Khác với generate_router_dataset.py (gọi Generator paraphrase, tốn thời
gian), file này sinh data bằng TEMPLATE COMPOSITION: ghép câu hỏi/order
thật (từ faq.csv, menu.csv) với các lớp từ đệm lịch sự đã biết (giống
fast_extract.py) + subject/context ngẫu nhiên. Nhãn {subject, action,
context} vì vậy CHÍNH XÁC 100% (không cần gán nhãn thủ công), sinh được
ngay lập tức không cần GPU/Generator.

Chạy:
    python3 generate_intent_dataset.py --n-per-core 12
Output: intent_dataset.jsonl — mỗi dòng {"text", "subject", "action", "context"}
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random

RNG = random.Random(42)

WRAPPERS = [
    ("", ""),
    ("Cho {subj} hỏi ", ""),
    ("Anh/chị ơi, ", ""),
    ("Excuse me, ", ""),
    ("Mình muốn biết ", ""),
    ("Xin hỏi ", ""),
    ("Không biết ", " vậy ạ?"),
    ("Làm ơn cho hỏi ", ""),
    ("", " cảm ơn shop."),
    ("", " (hỏi giúp bạn mình)"),
    ("", " ạ."),
    ("", " nhé."),
]

SUBJECTS = ["anh", "chị", "em", "mình", "tôi", "khách"]

TIME_CONTEXTS = [
    "ngày mai lúc 7h", "tối nay", "cuối tuần này", "sáng mai 8h30",
    "chiều nay lúc 3h", "", "", "",  # phần lớn không có ngữ cảnh thời gian
]


def load_faq_cores(csv_path: str) -> list[str]:
    """Nhóm các câu theo 'answer' (mỗi nhóm = 1 câu hỏi gốc + 10 biến thể từ
    đệm) rồi lấy câu NGẮN NHẤT trong nhóm làm core — biến thể có từ đệm luôn
    dài hơn bản gốc nên cách này chắc chắn đúng, không cần đoán regex."""
    groups: dict[str, list[str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row["question"].strip()
            groups.setdefault(row["answer"], []).append(q)
    cores = [min(qs, key=len) for qs in groups.values()]
    return cores


def load_menu_items(csv_path: str) -> list[tuple[str, str]]:
    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["name"].split("(")[0].strip()
            size = row.get("size", "").strip()
            items.append((name, size))
    # loại trùng
    seen = set()
    out = []
    for name, size in items:
        if name not in seen:
            seen.add(name)
            out.append((name, size))
    return out


CONSULTANT_CORES = [
    "tư vấn giúp món gì mát mát cho mùa hè",
    "có món nào ngọt mà không quá béo không",
    "gợi ý món nào đang hot không",
    "món nào phù hợp cho người mới uống cà phê",
    "có món nào ít đường không",
    "nên chọn size nào cho vừa uống",
    "món nào bán chạy nhất quán",
    "có combo nào tiết kiệm không",
]


def make_order_core(item: str, size: str) -> tuple[str, str]:
    qty = RNG.choice(["1", "2", "3"])
    size_txt = f" size {size}" if size and size != "Freesize" else ""
    action = f"đặt {qty} ly {item.lower()}{size_txt}"
    return action, qty


def apply_wrapper(core: str, subject: str) -> str:
    prefix, suffix = RNG.choice(WRAPPERS)
    prefix_filled = prefix.format(subj=subject) if "{subj}" in prefix else prefix
    text = f"{prefix_filled}{core}{suffix}"
    return text[0].upper() + text[1:] if text else text


def build_faq_examples(cores: list[str], n_per_core: int) -> list[dict]:
    examples = []
    for core in cores:
        for _ in range(n_per_core):
            subject = RNG.choice(SUBJECTS)
            text = apply_wrapper(core, subject)
            examples.append({
                "text": text,
                "subject": subject if RNG.random() > 0.4 else "khách",
                "action": core.rstrip("?").strip(),
                "context": "",
            })
    return examples


def build_order_examples(items: list[tuple[str, str]], n_per_item: int) -> list[dict]:
    """context CHỈ được gán nhãn khi nó THẬT SỰ xuất hiện trong text — nếu
    không, model học sai (nhãn không khớp input)."""
    examples = []
    for name, size in items:
        for _ in range(n_per_item):
            core, qty = make_order_core(name, size)
            subject = RNG.choice(["anh", "chị", "em"])
            context = RNG.choice(["", "", "", "mang đi", "uống tại chỗ", "ít đá", "không đường"])
            suffix = RNG.choice(["", " nhé", " giúp em với", " ạ"])
            if context:
                text = f"Cho {subject} {core}, {context}{suffix}."
            else:
                text = f"Cho {subject} {core}{suffix}."
            examples.append({
                "text": text,
                "subject": subject,
                "action": core,
                "context": context,
            })
    return examples


def build_consultant_examples(cores: list[str], n_per_core: int) -> list[dict]:
    examples = []
    for core in cores:
        for _ in range(n_per_core):
            subject = RNG.choice(SUBJECTS)
            text = apply_wrapper(core, subject)
            examples.append({
                "text": text,
                "subject": subject if RNG.random() > 0.4 else "khách",
                "action": core,
                "context": "",
            })
    return examples


def build_booking_examples(n: int) -> list[dict]:
    """Câu có ngữ cảnh thời gian rõ ràng (đặt bàn) — quan trọng để học {context}."""
    examples = []
    base_actions = ["đặt bàn tiệc sinh nhật", "đặt bàn cho nhóm 5 người", "đặt bàn trước"]
    for _ in range(n):
        subject = RNG.choice(["anh", "chị", "em"])
        action = RNG.choice(base_actions)
        time_ctx = RNG.choice([t for t in TIME_CONTEXTS if t])
        suffix = RNG.choice([" em nhé", " giúp em với", " nhé"])
        text = f"Cho {subject} {action} vào {time_ctx}{suffix}"
        text = text[0].upper() + text[1:]
        examples.append({
            "text": text,
            "subject": subject,
            "action": action,
            "context": time_ctx,
        })
    return examples


def dedup(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            out.append(r)
    return out


def main(n_per_core: int, out_path: str):
    here = os.path.dirname(__file__)
    faq_csv = os.path.join(here, "c2_data", "faq.csv")
    menu_csv = os.path.join(here, "c2_data", "menu.csv")

    faq_cores = load_faq_cores(faq_csv)
    menu_items = load_menu_items(menu_csv)
    print(f"FAQ cores (không từ đệm): {len(faq_cores)}")
    print(f"Menu items: {len(menu_items)}")

    rows = []
    rows += build_faq_examples(faq_cores, n_per_core)
    rows += build_order_examples(menu_items, max(2, n_per_core // 3))
    rows += build_consultant_examples(CONSULTANT_CORES, n_per_core)
    rows += build_booking_examples(n_per_core * 4)

    rows = dedup(rows)
    RNG.shuffle(rows)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Đã sinh {len(rows)} mẫu (đã loại trùng) -> {out_path}")
    print("Ví dụ 5 mẫu đầu:")
    for r in rows[:5]:
        print(" ", json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-core", type=int, default=12)
    parser.add_argument("--out", type=str, default="intent_dataset.jsonl")
    args = parser.parse_args()
    main(args.n_per_core, args.out)