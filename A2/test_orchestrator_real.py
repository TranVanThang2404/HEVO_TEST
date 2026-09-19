"""
Test full pipeline A2 với TOÀN BỘ component thật: SGLangRouterClient (Router
SGLang port 30000) + Neo4jKnowledgeBase (Graph RAG B1) + SGLangGeneratorClient
(Generator SGLang port 30001). Không còn Mock nào — đây là bằng chứng end-to-end
thật cho báo cáo/demo video.
"""
import asyncio
import time

import orchestrator as orch_module
from orchestrator import MultiAgentOrchestrator
from sglang_router_client import SGLangRouterClient
from request_queue import RequestQueue
from session_store import SessionStore


async def main():
    print("=== TEST FULL PIPELINE THẬT (Router + Neo4j RAG + Generator) ===\n")
    store = SessionStore(summarize_fn=orch_module.summarize_history)
    await store.start_background_cleanup()
    queue = RequestQueue(max_concurrency=3)
    bot = MultiAgentOrchestrator(SGLangRouterClient(), store, queue)

    session_id = "real-test-1"
    conversation = [
        "Cho anh 1 ly bạc xỉu size M",
        "Wifi quán tên gì vậy em?",
        "Có món nào ngọt mà không quá béo không?",
        "Mấy giờ quán đóng cửa?",
    ]
    for turn in conversation:
        t0 = time.perf_counter()
        result = await bot.handle_message(session_id, turn)
        latency = (time.perf_counter() - t0) * 1000
        print(f"User: {turn}")
        print(f"  -> intent={result['intent']:<10} agent={result['agent']:<10} ({latency:.0f}ms)")
        print(f"  Reply: {result['reply']}\n")

    _, history = store.get_context(session_id)
    print(f"Số turn lưu trong session: {len(history)}")
    await store.stop_background_cleanup()


if __name__ == "__main__":
    asyncio.run(main())
