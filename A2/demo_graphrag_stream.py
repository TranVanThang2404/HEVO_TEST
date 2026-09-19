# -*- coding: utf-8 -*-
"""
Demo Phần B — GraphRAG retrieval (Neo4j hybrid search + cache) nối tiếp
SSE streaming generation qua SGLang Generator, đo TTFT thật.

Khác api_server.py (không gọi GraphRAG) — script này chứng minh 2 phần
của B nối được với nhau: retrieval trước, rồi stream câu trả lời có
dùng context vừa lấy.

Đặt vào /mnt/d/HEVO/A2 (cùng thư mục orchestrator.py, knowledge_base.py,
neo4j_knowledge_base.py, cache_layer.py) rồi chạy:
    python3 demo_graphrag_stream.py

Cần: Neo4j đang chạy (docker start highlands-neo4j) + SGLang Generator
server đang chạy ở port 30001 (Terminal 1, đừng tắt).
"""
import asyncio
import time
import json
import httpx

from neo4j_knowledge_base import Neo4jKnowledgeBase
from cache_layer import GraphCache, cached_graph_search

SGLANG_GENERATOR_URL = "http://127.0.0.1:30001/v1/chat/completions"

QUERY = "Cho em hỏi cà phê sữa đá size L giá bao nhiêu vậy?"


async def retrieve_context(query: str, top_k: int = 3):
    neo4j_kb = Neo4jKnowledgeBase()
    cache = GraphCache()

    print(f"[GraphRAG] Truy vấn: '{query}'")
    t0 = time.perf_counter()
    results = await cached_graph_search(neo4j_kb, cache, query, top_k=top_k)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[GraphRAG] Lấy được {len(results)} chunk trong {elapsed:.1f}ms\n")

    for i, r in enumerate(results, 1):
        print(f"  [{i}] (raw) {r!r}")
    print()
    return results


def extract_text(r):
    """Cố gắng lấy field text từ nhiều dạng schema có thể có (dict hoặc object)."""
    if isinstance(r, dict):
        return r.get("text") or r.get("content") or str(r)
    return getattr(r, "text", None) or getattr(r, "content", None) or str(r)


def build_context_block(results) -> str:
    return "\n".join(f"- {extract_text(r)}" for r in results)


async def stream_answer(query: str, context_block: str):
    system_prompt = (
        "Bạn là nhân viên tư vấn quán cà phê Highlands. Dựa vào THÔNG TIN DƯỚI ĐÂY "
        "(trích từ menu/FAQ của quán, lấy qua GraphRAG) để trả lời khách hàng ngắn "
        "gọn, chính xác, thân thiện, bằng tiếng Việt. Nếu thông tin không có, đừng bịa.\n\n"
        f"THÔNG TIN:\n{context_block}"
    )
    payload = {
        "model": "generator",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "stream": True,
        "max_tokens": 256,
    }

    print("[Generator] Đang gửi request streaming (có context GraphRAG)...")
    t_start = time.perf_counter()
    first_token_time = None
    full_reply = ""

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", SGLANG_GENERATOR_URL, json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        ttft_ms = (first_token_time - t_start) * 1000
                        print(f"\n[TTFT] {ttft_ms:.1f} ms  (thời gian tới token đầu tiên)\n")
                        print("🤖 Trả lời (đang stream): ", end="", flush=True)
                    print(delta, end="", flush=True)
                    full_reply += delta

    total_ms = (time.perf_counter() - t_start) * 1000
    print(f"\n\n[Hoàn tất] Tổng thời gian: {total_ms:.1f} ms | Độ dài trả lời: {len(full_reply)} ký tự")


async def main():
    print("=" * 70)
    print("DEMO PHẦN B — GraphRAG Retrieval nối tiếp SSE Streaming (TTFT thật)")
    print("=" * 70)
    print()

    results = await retrieve_context(QUERY)
    context_block = build_context_block(results)
    await stream_answer(QUERY, context_block)

    print("\n" + "=" * 70)
    print("KẾT THÚC DEMO PHẦN B")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())