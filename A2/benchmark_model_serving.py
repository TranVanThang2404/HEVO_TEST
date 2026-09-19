"""
B2.4 (phần 1) — Benchmark tầng LLM SERVING (SGLang Generator), đo:
  1. TTFT (Time-To-First-Token)   — mục tiêu đề bài: <=200ms (Đạt) / <=70ms (Xuất sắc)
  2. Throughput (tokens/s)        — đo dưới tải đồng thời (concurrent requests)

Yêu cầu: Generator server (SGLang) đang chạy ở port 30001, model ĐÃ fix (không
còn gibberish) — nếu chưa fix, số liệu vẫn đo được (SGLang vẫn sinh đúng số
lượng token, chỉ là nội dung sai) nhưng KHÔNG nên dùng để báo cáo chính thức.

Chạy:
    python3 benchmark_model_serving.py --concurrency 1 --n 10
    python3 benchmark_model_serving.py --concurrency 5 --n 20   # đo throughput dưới tải
"""
import argparse
import asyncio
import json
import time

import httpx

GENERATOR_URL = "http://127.0.0.1:30001/v1/chat/completions"

TEST_QUERIES = [
    "Cho anh 1 ly cà phê sữa đá size L mang đi nhé.",
    "Có món nào ngọt mà không quá béo không?",
    "Wifi pass là gì vậy em?",
    "Quán mở cửa mấy giờ vậy?",
    "Trà đào cam sả giá bao nhiêu?",
    "Em ơi cho chị hỏi có chỗ đậu xe không?",
    "Tôi muốn đặt bàn cho 4 người lúc 7h tối nay.",
    "Có ưu đãi gì cho sinh viên không shop?",
    "Cho anh 1 phần bánh mì que và 1 ly bạc xỉu.",
    "Tổng đơn của em bao nhiêu tiền vậy ạ?",
]


async def stream_one(client: httpx.AsyncClient, query: str) -> dict:
    payload = {
        "model": "generator",
        "messages": [
            {"role": "system", "content": "Bạn là nhân viên tư vấn quán cà phê Highlands, trả lời ngắn gọn."},
            {"role": "user", "content": query},
        ],
        "stream": True,
        "max_tokens": 100,
    }
    t0 = time.perf_counter()
    first_token_t = None
    n_tokens = 0
    text = ""

    async with client.stream("POST", GENERATOR_URL, json=payload, timeout=30.0) as resp:
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                if first_token_t is None:
                    first_token_t = time.perf_counter()
                n_tokens += 1  # xấp xỉ: SGLang stream theo token thật, mỗi chunk ~1 token
                text += delta

    t_end = time.perf_counter()
    return {
        "ttft_ms": (first_token_t - t0) * 1000 if first_token_t else None,
        "total_ms": (t_end - t0) * 1000,
        "n_tokens": n_tokens,
        "text": text,
    }


async def run_load_test(concurrency: int, n_requests: int):
    queries = (TEST_QUERIES * ((n_requests // len(TEST_QUERIES)) + 1))[:n_requests]
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def worker(client, q):
        async with sem:
            r = await stream_one(client, q)
            results.append(r)

    wall_t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[worker(client, q) for q in queries])
    wall_time_s = time.perf_counter() - wall_t0

    return results, wall_time_s


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    idx = min(int(len(s) * p) , len(s) - 1)
    return s[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    print(f"[Benchmark] concurrency={args.concurrency}, n_requests={args.n}")
    results, wall_time_s = asyncio.run(run_load_test(args.concurrency, args.n))

    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    totals = [r["total_ms"] for r in results]
    total_tokens = sum(r["n_tokens"] for r in results)

    print("\n" + "=" * 60)
    print(f"KẾT QUẢ (n={len(results)} request, concurrency={args.concurrency})")
    print("=" * 60)
    print(f"TTFT      — P50: {percentile(ttfts, 0.5):.1f} ms | "
          f"P95: {percentile(ttfts, 0.95):.1f} ms | avg: {sum(ttfts)/len(ttfts):.1f} ms")
    print(f"Total lat — P50: {percentile(totals, 0.5):.1f} ms | "
          f"P95: {percentile(totals, 0.95):.1f} ms | avg: {sum(totals)/len(totals):.1f} ms")
    print(f"Throughput toàn hệ thống: {total_tokens / wall_time_s:.1f} tokens/s "
          f"(tổng {total_tokens} token / {wall_time_s:.2f}s tường, concurrency={args.concurrency})")

    print("\nĐối chiếu mục tiêu đề bài (đo trên RTX 4050, khác RTX 3060 giả định -> ghi rõ trong báo cáo):")
    p50_ttft = percentile(ttfts, 0.5)
    for label, budget in [("Đạt (<=200ms)", 200), ("Xuất sắc (<=70ms)", 70)]:
        status = "PASS" if p50_ttft <= budget else "FAIL"
        print(f"  TTFT P50 vs {label}: {status}")


if __name__ == "__main__":
    main()