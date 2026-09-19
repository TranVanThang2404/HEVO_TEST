"""
B2.4 — Benchmark Throughput: số tokens/giây khi Generator serving ĐỒNG THỜI
nhiều request cùng lúc (không phải tuần tự như benchmark_ttft.py).

Chạy:
    python3 benchmark_throughput.py --concurrency 8 --max-tokens 128
"""
from __future__ import annotations

import argparse
import asyncio
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


async def one_request(client: httpx.AsyncClient, prompt: str, max_tokens: int) -> tuple[float, int]:
    """Trả về (latency_ms, completion_tokens) của 1 request (non-streaming,
    lấy completion_tokens chính xác từ usage của SGLang trả về)."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Bạn là trợ lý quán cà phê."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
    }
    t0 = time.perf_counter()
    resp = await client.post(GENERATOR_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()
    latency_ms = (time.perf_counter() - t0) * 1000
    completion_tokens = data["usage"]["completion_tokens"]
    return latency_ms, completion_tokens


async def main(concurrency: int, max_tokens: int):
    async with httpx.AsyncClient(timeout=60.0) as client:
        # warm-up
        await one_request(client, TEST_PROMPTS[0], max_tokens)

        prompts = [TEST_PROMPTS[i % len(TEST_PROMPTS)] for i in range(concurrency)]

        t0 = time.perf_counter()
        results = await asyncio.gather(*[one_request(client, p, max_tokens) for p in prompts])
        wall_time_s = time.perf_counter() - t0

        total_tokens = sum(tok for _, tok in results)
        per_req_latencies = [lat for lat, _ in results]

        print(f"=== THROUGHPUT (concurrency={concurrency}, max_tokens={max_tokens}) ===")
        print(f"  Wall time (tất cả {concurrency} request chạy song song): {wall_time_s:.2f} s")
        print(f"  Tổng completion_tokens sinh ra           : {total_tokens} tokens")
        print(f"  Throughput hệ thống (aggregate)           : {total_tokens / wall_time_s:.1f} tokens/s")
        print(f"  Latency từng request (ms)  : min={min(per_req_latencies):.0f} "
              f"avg={sum(per_req_latencies)/len(per_req_latencies):.0f} max={max(per_req_latencies):.0f}")
        print("\nLƯU Ý: đo trên RTX 4050 6GB, khác RTX 3060 12GB giả định của đề bài;")
        print(f"        --max-running-requests khi launch Generator giới hạn số request")
        print(f"        xử lý song song thật sự trong SGLang scheduler.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    asyncio.run(main(args.concurrency, args.max_tokens))