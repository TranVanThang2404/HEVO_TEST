# -*- coding: utf-8 -*-
"""
Benchmark thật cho Phần C2 (C2.1 Intent Extraction + C2.2 Cache Pipeline)
trên dữ liệu FAQ thật (faq.csv): 16 câu hỏi gốc x 10 biến thể ngữ cảnh
= 160 câu, đúng dạng "câu hỏi lặp lại với ngữ cảnh khác nhau" mà đề bài
mô tả (trang 12, bối cảnh C2).

GHI CHÚ PHẠM VI (trung thực): đề bài yêu cầu benchmark trên "bộ 500 câu
hỏi F&B thực tế" — bản này benchmark trên 160 câu FAQ thật có sẵn (chưa
đủ 500). Kết quả vẫn là bằng chứng thật, chỉ chưa đủ quy mô mẫu theo đề.

Đặt vào /mnt/d/HEVO/A2 (cùng thư mục intent_extractor.py, semantic_cache_c2.py)
và đặt file faq.csv vào /mnt/d/HEVO/A2/c2_data/faq.csv (hoặc sửa CSV_PATH),
rồi chạy:
    python3 c2_benchmark.py
"""
import csv
import os
import statistics
import time

from semantic_cache_c2 import SemanticCacheC2

CSV_PATH = os.path.join(os.path.dirname(__file__), "c2_data", "faq.csv")


def load_faq_rows(csv_path: str):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def make_agent_fn(gold_answer: str):
    """Giả lập gọi FAQAgent thật: trả về đúng câu trả lời gốc trong faq.csv
    (đóng vai trò 'response_template thật' mà Agent sẽ trả về khi cache miss).
    Đây LÀ dữ liệu thật của quán, không phải bịa."""

    def _agent_fn(action_text: str, context_text: str) -> str:
        return gold_answer

    return _agent_fn


def main():
    print("=" * 70)
    print("BENCHMARK PHẦN C2 — Intent Extraction + Semantic Cache (dữ liệu FAQ thật)")
    print("=" * 70)

    rows = load_faq_rows(CSV_PATH)
    print(f"\nĐã nạp {len(rows)} câu hỏi thật từ {CSV_PATH}\n")

    cache = SemanticCacheC2(similarity_threshold=0.92)

    results = []
    t_start = time.perf_counter()
    for i, row in enumerate(rows, start=1):
        query = row["question"]
        gold_answer = row["answer"]
        category = row["category"]

        result = cache.process(query, make_agent_fn(gold_answer))
        result["category"] = category
        result["query"] = query
        result["gold_answer"] = gold_answer
        results.append(result)

        tag = "HIT " if result["cache_hit"] else "MISS"
        print(
            f"[{i:3d}/{len(rows)}] {tag}  sim={result['similarity']:.3f}  "
            f"total={result['latency_ms']:6.1f}ms  extract={result['extract_latency_ms']:6.1f}ms  "
            f"action='{result['action']}'"
        )
    total_elapsed_s = time.perf_counter() - t_start

    # ----- Thống kê -----
    hit_results = [r for r in results if r["cache_hit"]]
    miss_results = [r for r in results if not r["cache_hit"]]

    print("\n" + "=" * 70)
    print("KẾT QUẢ TỔNG HỢP")
    print("=" * 70)
    print(f"Tổng số câu           : {len(results)}")
    print(f"Cache HIT             : {len(hit_results)}")
    print(f"Cache MISS            : {len(miss_results)}")
    print(f"Hit rate              : {cache.hit_rate * 100:.1f}%  (yêu cầu đề bài: >= 60%)")

    if hit_results:
        hit_latencies = [r["latency_ms"] for r in hit_results]
        print(f"\nLatency cache-HIT (ms): min={min(hit_latencies):.1f}  "
              f"median={statistics.median(hit_latencies):.1f}  max={max(hit_latencies):.1f}")
        print(f"  (yêu cầu đề bài: <= 100ms, KHÔNG qua LLM Generator lớn — "
              f"lưu ý bước Extract vẫn dùng SLM 0.5B nên có thể > 100ms, xem ghi chú)")
        over_100ms = sum(1 for x in hit_latencies if x > 100)
        print(f"  Số lần HIT vượt quá 100ms: {over_100ms}/{len(hit_latencies)}")

    if miss_results:
        miss_latencies = [r["latency_ms"] for r in miss_results]
        print(f"\nLatency cache-MISS (ms): min={min(miss_latencies):.1f}  "
              f"median={statistics.median(miss_latencies):.1f}  max={max(miss_latencies):.1f}")

    print(f"\nTổng thời gian chạy toàn bộ benchmark: {total_elapsed_s:.1f}s")

    # ----- Test cache invalidation (C3 nhắc tới, hữu ích cho C2 luôn) -----
    print("\n" + "=" * 70)
    print("TEST CACHE INVALIDATION (khi menu/FAQ thay đổi)")
    print("=" * 70)
    sample_query = rows[1]["question"]  # 1 biến thể của câu đầu tiên
    result_before = cache.process(sample_query, make_agent_fn(rows[0]["answer"]))
    print(f"Trước invalidate — cache_hit = {result_before['cache_hit']} (kỳ vọng: True, vì đã cache ở vòng trên)")
    cache.invalidate_all()
    result_after = cache.process(sample_query, make_agent_fn(rows[0]["answer"]))
    print(f"Sau invalidate_all() — cache_hit = {result_after['cache_hit']} (kỳ vọng: False, vì cache đã bị xoá sạch)")

    print("\n" + "=" * 70)
    print("HOÀN TẤT BENCHMARK C2")
    print("=" * 70)


if __name__ == "__main__":
    main()