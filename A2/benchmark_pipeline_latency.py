"""
B2.4 (phần 2) — Benchmark TOTAL PIPELINE LATENCY + xác nhận tiêu chí nghiệm thu B2.
"""
from __future__ import annotations

import argparse
import asyncio
import time
import uuid

from orchestrator import MockRouterClient, MultiAgentOrchestrator
from request_queue import RequestQueue
from session_store import SessionStore

TEST_MESSAGES = [
    "Cho anh 1 ly cà phê sữa đá size L mang đi nhé.",
    "Có món nào ngọt mà không quá béo không?",
    "Wifi pass là gì vậy em?",
    "Quán mở cửa mấy giờ vậy?",
    "Trà đào cam sả giá bao nhiêu?",
]

CACHE_TEST_QUERY = "Quán có chỗ gửi xe không?"


async def run_one(orchestrator: MultiAgentOrchestrator, message: str, session_id: str | None = None) -> float:
    session_id = session_id or f"bench-{uuid.uuid4().hex[:8]}"
    t0 = time.perf_counter()
    _ = await orchestrator.handle_message(session_id, message)
    return (time.perf_counter() - t0) * 1000


def _stats(latencies_ms: list[float]) -> dict:
    s = sorted(latencies_ms)
    n = len(s)
    return {
        "n": n,
        "avg_ms": sum(s) / n,
        "p50_ms": s[n // 2],
        "p95_ms": s[min(int(n * 0.95), n - 1)],
        "min_ms": s[0],
        "max_ms": s[-1],
    }


async def measure_total_pipeline_latency(orchestrator: MultiAgentOrchestrator, n: int) -> list[float]:
    messages = (TEST_MESSAGES * ((n // len(TEST_MESSAGES)) + 1))[:n]
    await run_one(orchestrator, messages[0])
    latencies = []
    for i, msg in enumerate(messages):
        lat = await run_one(orchestrator, msg)
        latencies.append(lat)
        print(f"  [{i + 1}/{n}] '{msg[:40]}...' -> {lat:.1f} ms")
    return latencies


async def measure_cache_speedup(orchestrator: MultiAgentOrchestrator) -> tuple[float, float]:
    lat1 = await run_one(orchestrator, CACHE_TEST_QUERY, session_id=f"cache-test-1-{uuid.uuid4().hex[:6]}")
    lat2 = await run_one(orchestrator, CACHE_TEST_QUERY, session_id=f"cache-test-2-{uuid.uuid4().hex[:6]}")
    return lat1, lat2


def _print_stats(title: str, s: dict, budgets: list[tuple[str, float]]):
    print(f"\n=== {title} (n={s['n']}) ===")
    print(f"  Avg : {s['avg_ms']:.1f} ms")
    print(f"  P50 : {s['p50_ms']:.1f} ms")
    print(f"  P95 : {s['p95_ms']:.1f} ms")
    print(f"  Min : {s['min_ms']:.1f} ms")
    print(f"  Max : {s['max_ms']:.1f} ms")
    for label, budget_ms in budgets:
        status = "PASS" if s["p50_ms"] <= budget_ms else "FAIL"
        print(f"  P50 vs {label} (<= {budget_ms:.0f}ms): {status}")


async def main(n: int):
    print("Khởi tạo Orchestrator...")
    orchestrator = MultiAgentOrchestrator(MockRouterClient(), SessionStore(), RequestQueue())

    # Warm-up: PHẢI dùng câu khớp keyword (faq/consultant) để thật sự ép load
    # embedding + reranker model trước khi đo, không rơi vào nhánh "ignore".
    await run_one(orchestrator, "Wifi pass là gì vậy em?")

    # ---- 2. Cache Speedup — đo khi CACHE_TEST_QUERY còn "sạch" ----
    print("=" * 70)
    print("2. CACHE SPEEDUP — tiêu chí nghiệm thu B2...")
    print("=" * 70)
    try:
        lat1, lat2 = await measure_cache_speedup(orchestrator)
        speedup_pct = (lat1 - lat2) / lat1 * 100 if lat1 > 0 else 0.0
        print(f"  Câu hỏi test: '{CACHE_TEST_QUERY}'")
        print(f"  Lần 1 (cache MISS thật) : {lat1:.1f} ms")
        print(f"  Lần 2 (cache HIT thật)  : {lat2:.1f} ms")
        print(f"  Nhanh hơn: {speedup_pct:.1f}%")
        cache_pass = lat2 < lat1 * 0.7
        print(f"  -> {'PASS' if cache_pass else 'FAIL'}")
        print(f"  Cache stats: {orchestrator.graph_cache.stats()}")
    except Exception as e:
        print(f"  [LỖI] {e}")

    # ---- 1. Total Pipeline Latency ----
    latencies = await measure_total_pipeline_latency(orchestrator, n)
    stats = _stats(latencies)
    _print_stats("1. TOTAL PIPELINE LATENCY", stats, [("Đạt", 1500), ("Xuất sắc", 800)])

    print("\n" + "=" * 70)
    print("LƯU Ý BÁO CÁO: đo trên RTX 4050 6GB, khác RTX 3060 12GB giả định của đề")
    print("bài -> ghi rõ trong báo cáo kỹ thuật (giống benchmark_model_serving.py).")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(main(args.n))