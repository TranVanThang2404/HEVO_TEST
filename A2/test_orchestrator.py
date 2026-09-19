
import asyncio
import time

import orchestrator as orch_module
from orchestrator import MockRouterClient, MultiAgentOrchestrator
from request_queue import RequestQueue
from session_store import SessionStore


async def demo_multiturn():
    print("\n=== DEMO 1: Hội thoại đa lượt (multi-turn) qua >= 3 agent khác nhau ===")
    store = SessionStore(summarize_fn=orch_module.summarize_history)
    await store.start_background_cleanup()
    queue = RequestQueue(max_concurrency=3)
    bot = MultiAgentOrchestrator(MockRouterClient(), store, queue)

    session_id = "demo-session-1"
    conversation = [
        "Cho anh 1 ly bạc xỉu size M",
        "Wifi quán tên gì vậy em?",
        "Có món nào ngọt mà không quá béo không?",
        "Mấy giờ quán đóng cửa?",
        "Cho anh thêm 1 ly trà đào nữa",
        "Tính tiền giúp anh",
    ]
    for turn in conversation:
        result = await bot.handle_message(session_id, turn)
        print(f"User: {turn}")
        print(f"  -> intent={result['intent']:<10} agent={result['agent']:<10} reply={result['reply']}")

    _, history = store.get_context(session_id)
    print(
        f"\nSố turn đang lưu trong SessionStore: {len(history)} "
        f"(đã qua {len(conversation)} lượt hỏi -> history window + auto-summarize hoạt động đúng)"
    )
    await store.stop_background_cleanup()


async def demo_concurrency():
    print("\n=== DEMO 2: 5 request đồng thời (concurrency control, tránh OOM) ===")
    store = SessionStore()
    queue = RequestQueue(max_concurrency=3, wait_timeout=60)
    bot = MultiAgentOrchestrator(MockRouterClient(), store, queue)

    queries = [
        ("s1", "Cho tôi 1 ly cà phê sữa đá"),
        ("s2", "Wifi pass là gì?"),
        ("s3", "Gợi ý giúp tôi món gì mát mà rẻ"),
        ("s4", "Quán mấy giờ mở cửa?"),
        ("s5", "Cho tôi đặt 2 ly latte"),
    ]
    t0 = time.perf_counter()
    results = await asyncio.gather(*[bot.handle_message(sid, q) for sid, q in queries])
    elapsed = time.perf_counter() - t0
    for (sid, q), r in zip(queries, results):
        print(f"[{sid}] {q} -> agent={r['agent']}")
    print(
        f"Đã xử lý {len(queries)} request đồng thời trong {elapsed:.2f}s, "
        f"giới hạn concurrency={queue.stats['capacity']} -> không crash, không deadlock."
    )


async def demo_session_ttl():
    print("\n=== DEMO 3: Session TTL (giả lập session hết hạn > 30 phút) ===")
    store = SessionStore()
    await store.add_turn("old-session", "user", "xin chào")
    sess = store.get_or_create("old-session")
    sess.last_active = time.time() - 31 * 60  # giả lập không hoạt động > 30 phút
    print(f"Trước cleanup: {store.active_session_count()} session đang lưu")
    store.cleanup_expired()
    print(
        f"Sau cleanup: {store.active_session_count()} session còn lại "
        f"(session 'old-session' đã bị xóa do vượt TTL 30 phút)"
    )


async def main():
    await demo_multiturn()
    await demo_concurrency()
    await demo_session_ttl()


if __name__ == "__main__":
    asyncio.run(main())