"""
B2.2 — Test/Benchmark SSE Streaming: xác nhận stream hoạt động đúng và
KHÔNG MẤT TOKEN (tiêu chí nghiệm thu B2).

Cách kiểm tra "không mất token": so sánh nội dung ghép lại từ luồng event
"token" (thô) với luồng event "clause" (được server dựng TỪ chính luồng
token qua ClauseBuffer) — nếu 2 bên lệch nhau (ngoài khoảng trắng bị strip
ở biên mỗi clause) nghĩa là có ký tự bị rớt ở tầng SSE/network.

Yêu cầu: api_server.py đã chạy (uvicorn api_server:app --port 8080).

Chạy:
    python3 test_sse_streaming.py --n 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time

import httpx

API_URL = "http://127.0.0.1:8080/v1/chat/stream"

TEST_MESSAGES = [
    "Cho anh 1 ly cà phê sữa đá size L mang đi nhé.",
    "Có món nào ngọt mà không quá béo không?",
    "Wifi pass là gì vậy em?",
    "Quán mở cửa mấy giờ vậy?",
]


def _parse_sse_blocks(raw_text: str) -> list[tuple[str | None, str]]:
    """Parse SSE thô thành list (event_name|None, data_str)."""
    blocks = raw_text.strip("\n").split("\n\n")
    parsed = []
    for block in blocks:
        if not block.strip():
            continue
        event_name = None
        data_str = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_str = line[len("data:"):].strip()
        if data_str is not None:
            parsed.append((event_name, data_str))
    return parsed


async def run_one_sse_test(client: httpx.AsyncClient, message: str, session_id: str) -> dict:
    t0 = time.perf_counter()
    ttft_ms = None
    token_events = []
    clause_events = []
    done_received = False
    done_is_raw_literal = False  # PHẢI là "[DONE]" thô, KHÔNG bọc JSON
    error = None

    payload = {"session_id": session_id, "message": message}

    async with client.stream("POST", API_URL, json=payload) as resp:
        resp.raise_for_status()
        buffer = ""
        async for chunk in resp.aiter_text():
            buffer += chunk
            # Xử lý từng block SSE hoàn chỉnh (kết thúc bằng \n\n) ngay khi có,
            # để test đúng bản chất TRUE STREAMING (không đợi đọc hết cả response).
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                if not block.strip():
                    continue

                # Sentinel [DONE]: PHẢI check literal string TRƯỚC khi json.loads,
                # đúng như docstring api_server.py yêu cầu.
                if block.strip() == "data: [DONE]":
                    done_received = True
                    done_is_raw_literal = True
                    continue

                event_name = None
                data_str = None
                for line in block.split("\n"):
                    if line.startswith("event:"):
                        event_name = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        data_str = line[len("data:"):].strip()

                if data_str is None:
                    continue

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError as e:
                    error = f"JSON parse lỗi ở block: {block!r} ({e})"
                    continue

                if event_name == "token":
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    token_events.append(data["delta"]["content"])
                elif event_name == "clause":
                    clause_events.append(data["clause"])
                elif event_name == "error":
                    error = data.get("error")

    total_ms = (time.perf_counter() - t0) * 1000
    text_from_tokens = "".join(token_events)
    text_from_clauses = " ".join(clause_events)  # mỗi clause đã bị strip() 2 đầu

    # So sánh KHÔNG phân biệt khoảng trắng thừa (chỉ quan tâm có mất KÝ TỰ THỰC không)
    norm_tokens = re.sub(r"\s+", " ", text_from_tokens).strip()
    norm_clauses = re.sub(r"\s+", " ", text_from_clauses).strip()
    consistent = norm_tokens == norm_clauses

    return {
        "message": message,
        "error": error,
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "n_token_events": len(token_events),
        "n_clause_events": len(clause_events),
        "done_received": done_received,
        "done_is_raw_literal": done_is_raw_literal,
        "consistent": consistent,
        "text_from_tokens": norm_tokens,
        "text_from_clauses": norm_clauses,
    }


async def main(n: int):
    messages = (TEST_MESSAGES * ((n // len(TEST_MESSAGES)) + 1))[:n]
    results = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, msg in enumerate(messages):
            r = await run_one_sse_test(client, msg, session_id=f"sse-test-{i}")
            results.append(r)
            status = "OK" if (r["done_received"] and r["done_is_raw_literal"]
                               and r["consistent"] and not r["error"]) else "FAIL"
            print(f"  [{i + 1}/{n}] '{msg[:35]}...' -> TTFT={r['ttft_ms']:.1f}ms "
                  f"total={r['total_ms']:.1f}ms tokens={r['n_token_events']} "
                  f"clauses={r['n_clause_events']} consistent={r['consistent']} "
                  f"[{status}]")
            if status == "FAIL":
                print(f"        error={r['error']}")
                if not r["consistent"]:
                    print(f"        TOKENS : {r['text_from_tokens'][:200]}")
                    print(f"        CLAUSES: {r['text_from_clauses'][:200]}")

    n_ok = sum(1 for r in results if r["done_received"] and r["done_is_raw_literal"]
               and r["consistent"] and not r["error"])
    print(f"\n=== TỔNG KẾT: {n_ok}/{len(results)} request PASS ===")
    print("Tiêu chí:")
    print(f"  - Nhận đủ sentinel 'data: [DONE]' literal (không bọc JSON): "
          f"{sum(1 for r in results if r['done_is_raw_literal'])}/{len(results)}")
    print(f"  - Nội dung token-stream khớp clause-stream (không mất ký tự): "
          f"{sum(1 for r in results if r['consistent'])}/{len(results)}")
    print(f"  - Không có lỗi giữa chừng: "
          f"{sum(1 for r in results if not r['error'])}/{len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(main(args.n))