"""
B2.4 — Benchmark TTFT (Time-To-First-Token) cho Generator (SGLang, port 30001).

Đo bằng cách gọi THẲNG /v1/chat/completions với stream=True (bỏ qua toàn bộ
orchestrator/Router/Neo4j — chỉ đo tốc độ sinh token đầu tiên của Generator,
đúng như đề bài mô tả "TTFT (Generator, RTX 3060)").

Chạy:
    python3 benchmark_ttft.py --n 15
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx

GENERATOR_URL = "http://127.0.0.1:30001/v1/chat/completions"
MODEL_NAME = "generator"

TEST_PROMPTS = [
    "Cho anh 1 ly cà phê sữa đá size L mang đi nhé.",
    "Có món nào ngọt mà không quá béo không?",
    "Wifi pass là gì vậy em?",
    "Quán mở cửa mấy giờ vậy?",
    "Trà đào cam sả giá bao nhiêu?",
]


async def measure_ttft_one(client: httpx.AsyncClient, prompt: str) -> float:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Bạn là trợ lý quán cà phê."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 128,
        "stream": True,
    }
    t0 = time.perf_counter()
    async with client.stream("POST", GENERATOR_URL, json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                return (time.perf_counter() - t0) * 1000
    return -1.0  # không nhận được token nào — báo lỗi rõ nếu xảy ra


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


async def main(n: int):
    prompts = (TEST_PROMPTS * ((n // len(TEST_PROMPTS)) + 1))[:n]
    async with httpx.AsyncClient(timeout=30.0) as client:
        # warm-up 1 lượt (không tính) — tránh lẫn chi phí khởi động CUDA graph lần đầu
        await measure_ttft_one(client, prompts[0])

        latencies = []
        for i, p in enumerate(prompts):
            lat = await measure_ttft_one(client, p)
            latencies.append(lat)
            print(f"  [{i + 1}/{n}] '{p[:40]}...' -> TTFT={lat:.1f} ms")

    stats = _stats(latencies)
    print(f"\n=== TTFT (Generator, n={stats['n']}) ===")
    print(f"  Avg : {stats['avg_ms']:.1f} ms")
    print(f"  P50 : {stats['p50_ms']:.1f} ms")
    print(f"  P95 : {stats['p95_ms']:.1f} ms")
    print(f"  Min : {stats['min_ms']:.1f} ms")
    print(f"  Max : {stats['max_ms']:.1f} ms")
    for label, budget_ms in [("Đạt", 200), ("Xuất sắc", 70)]:
        status = "PASS" if stats["p50_ms"] <= budget_ms else "FAIL"
        print(f"  P50 vs {label} (<= {budget_ms:.0f}ms): {status}")
    print("\nLƯU Ý: đo trên RTX 4050 6GB, khác RTX 3060 12GB giả định của đề bài.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15)
    args = parser.parse_args()
    asyncio.run(main(args.n))