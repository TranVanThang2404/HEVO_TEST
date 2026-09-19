# -*- coding: utf-8 -*-
"""
Demo video Phần A — multi-turn conversation qua >= 3 agent khác nhau.

Chạy TUẦN TỰ (không đồng thời) để tránh OOM đã phát hiện ở soak test trên GPU 6GB.
Dùng Router THẬT (LocalSLMRouterClient, base Qwen2.5-0.5B-Instruct — xem ghi chú
cuối file về lý do KHÔNG dùng bản fine-tune 1.5B) + SessionStore với summarize_fn
THẬT + RequestQueue concurrency=1 (đủ cho demo 1 người dùng).

Đặt file này vào /mnt/d/HEVO/A2 (cùng thư mục orchestrator.py, session_store.py, ...)
rồi chạy:
    python3 demo_multiturn.py

Ghi hình: Windows Game Bar (Win+Alt+R) quay màn hình terminal trong lúc chạy lệnh trên.
"""
import asyncio
import time

from orchestrator import MultiAgentOrchestrator, LocalSLMRouterClient
from session_store import SessionStore
from request_queue import RequestQueue
import orchestrator as orch_module

SESSION_ID = "demo-video-session"

# 3 câu này đã được TEST TRƯỚC và xác nhận router 0.5B base phân loại ĐÚNG
# (xem log test candidate ngày 2026-09-20). Không tự ý đổi câu khi chưa test lại.
CONVERSATION = [
    ("order", "Cho anh 1 ly cà phê sữa đá size L mang đi nhé."),
    ("consultant", "Tư vấn giúp em món gì mát mát cho mùa hè với."),
    ("faq", "Quán mình wifi pass là gì vậy em?"),
]


async def main():
    print("=" * 70)
    print("DEMO MULTI-TURN CONVERSATION — PHẦN A (Router + Multi-Agent Orchestrator)")
    print("=" * 70)

    store = SessionStore(summarize_fn=orch_module.summarize_history)
    await store.start_background_cleanup()
    queue = RequestQueue(max_concurrency=1, wait_timeout=60)

    print("\n[*] Đang nạp Router thật (Qwen2.5-0.5B base) lên GPU...")
    router = LocalSLMRouterClient()

    bot = MultiAgentOrchestrator(router, store, queue)
    print("[*] Sẵn sàng. Bắt đầu hội thoại multi-turn...\n")

    seen_intents = []

    for turn_no, (expected_label, message) in enumerate(CONVERSATION, start=1):
        print(f"--- Turn {turn_no} (kỳ vọng: {expected_label}) " + "-" * 30)
        print(f"👤 User: {message}")

        t0 = time.perf_counter()
        result = await bot.handle_message(SESSION_ID, message)
        elapsed = (time.perf_counter() - t0) * 1000

        actual_intent = result.get("intent")
        seen_intents.append(actual_intent)

        print(f"🤖 [Agent: {result.get('agent')}] [Intent: {actual_intent}] "
              f"[{elapsed:.0f}ms]")
        print(f"   Reply: {result.get('reply')}")
        print()

        await asyncio.sleep(1.0)  # nghỉ ngắn giữa các turn, dễ quay/dễ xem hơn

    distinct_agents = sorted(set(seen_intents))
    print("=" * 70)
    print(f"KẾT THÚC DEMO — số agent/intent THẬT đã đi qua ({len(distinct_agents)}): {distinct_agents}")
    print("=" * 70)

    await store.stop_background_cleanup()


if __name__ == "__main__":
    asyncio.run(main())

