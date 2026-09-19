import json

with open("router_dataset.jsonl", encoding="utf-8") as f:
    records = [json.loads(l) for l in f if l.strip()]

review_by_idx = {}
with open("router_dataset_review.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        review_by_idx[r["index"]] = r

n_changed = 0
for i, rec in enumerate(records):
    if not rec.get("is_hard"):
        continue  # CHỈ áp dụng sửa cho phần hard, bỏ qua normal (bị judge yếu làm nhiễu)
    rv = review_by_idx.get(i)
    if rv and rv["confidence"] in ("high", "medium") and rv["judged_label"] != rv["original_label"]:
        rec["intent"] = rv["judged_label"]
        n_changed += 1

with open("router_dataset_cleaned_hardonly_v2.jsonl", "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Đã tái tạo dataset chỉ-sạch-hard: {n_changed} mẫu đổi nhãn (kỳ vọng ~211).")