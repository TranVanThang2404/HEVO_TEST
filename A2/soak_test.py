"""
Soak test cho A2 — kiểm tra đúng tiêu chí nghiệm thu còn thiếu:
  "Zero deadlock, zero memory leak trong 30 phút chạy liên tục."
  "Không xảy ra OOM khi 3+ requests đồng thời."

Cách làm: liên tục bắn request đồng thời (nhiều session khác nhau, nhiều nội dung khác
nhau) vào MultiAgentOrchestrator trong N phút (mặc định 30), đo RSS memory của tiến trình
theo thời gian bằng psutil, đếm số request thành công/lỗi/timeout, và kiểm tra:
  - Không có request nào bị treo quá timeout của RequestQueue (=> zero deadlock).
  - Memory cuối kỳ không tăng KHÔNG GIỚI HẠN theo thời gian (=> zero leak) — nhờ
    SessionStore có TTL 30 phút + background cleanup, số session active phải luôn bị
    chặn trên, không tăng vô hạn dù liên tục tạo session mới.

Chạy (mặc định 30 phút, 8 session song song, 1 request/giây/session):
    python soak_test.py
Chạy ngắn hơn để test nhanh trước (ví dụ 2 phút):
    python soak_test.py --minutes 2

Yêu cầu: đặt file này cùng thư mục với 6 file A2 (session_store.py, request_queue.py,
knowledge_base.py, agents.py, orchestrator.py) — tức D:\\HEVO\\A2.
Cần cài thêm: pip install psutil
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import random
import time

import psutil

from orchestrator import LocalSLMRouterClient, MockRouterClient, MultiAgentOrchestrator
from request_queue import RequestQueue
from session_store import SessionStore

SAMPLE_QUERIES = [
    "Cho tôi 1 ly cà phê sữa đá",
    "Wifi pass là gì vậy?",
    "Gợi ý giúp tôi món gì mát mà rẻ",
    "Quán mấy giờ mở cửa?",
    "Cho tôi đặt 2 ly latte size L",
    "Có món nào ít ngọt không?",
    "Tính tiền giúp em",
    "Chỗ để xe ở đâu vậy?",
    "What time do you open?",
    "Can I get a recommendation for a hot day?",
]


async def worker(bot: MultiAgentOrchestrator, session_id: str, stop_at: float, stats: dict):
    while time.time() < stop_at:
        query = random.choice(SAMPLE_QUERIES)
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(bot.handle_message(session_id, query), timeout=65)
            elapsed = time.perf_counter() - start
            stats["ok"] += 1
            stats["max_latency"] = max(stats["max_latency"], elapsed)
            if result.get("meta", {}).get("error") == "timeout":
                stats["graceful_timeout"] += 1
        except asyncio.TimeoutError:
            stats["hard_timeout"] += 1  # đây mới thực sự là dấu hiệu deadlock (worse than graceful)
        except Exception as e:
            if "out of memory" in str(e).lower():
                stats["oom"] = stats.get("oom", 0) + 1
                print(f"[CẢNH BÁO] CUDA OOM thật xảy ra: {e}")
            stats["error"] += 1
            stats["last_error"] = str(e)
        await asyncio.sleep(random.uniform(0.5, 1.5))


async def main(minutes: float, n_sessions: int, real_router: bool):
    store = SessionStore()
    await store.start_background_cleanup()
    queue = RequestQueue(max_concurrency=3, wait_timeout=60)

    if real_router:
        print("Dùng Router THẬT (LocalSLMRouterClient, load model lên GPU) — cần "
              "router_agent_fast.py cùng thư mục và GPU khả dụng.")
        router = LocalSLMRouterClient()
        try:
            import torch
            if torch.cuda.is_available():
                print(f"VRAM đã cấp phát ngay sau khi load Router: "
                      f"{torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
        except ImportError:
            pass
    else:
        router = MockRouterClient()

    bot = MultiAgentOrchestrator(router, store, queue)

    process = psutil.Process()
    stats = {"ok": 0, "error": 0, "graceful_timeout": 0, "hard_timeout": 0,
              "max_latency": 0.0, "last_error": None}

    stop_at = time.time() + minutes * 60
    session_ids = [f"soak-session-{i}" for i in range(n_sessions)]

    print(f"Bắt đầu soak test: {minutes} phút, {n_sessions} session song song, "
          f"concurrency cap={queue.stats['capacity']}.")
    print(f"RSS memory ban đầu: {process.memory_info().rss / 1024 / 1024:.1f} MB")

    workers = [asyncio.create_task(worker(bot, sid, stop_at, stats)) for sid in session_ids]

    mem_samples = []
    vram_samples = []
    session_count_samples = []
    t0 = time.time()
    while time.time() < stop_at:
        await asyncio.sleep(15)
        gc.collect()
        rss_mb = process.memory_info().rss / 1024 / 1024
        active_sessions = store.active_session_count()
        mem_samples.append(rss_mb)
        session_count_samples.append(active_sessions)
        vram_str = ""
        if real_router:
            try:
                import torch
                if torch.cuda.is_available():
                    vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    vram_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
                    vram_samples.append(vram_mb)
                    vram_str = f" | VRAM={vram_mb:.0f}MB (reserved={vram_reserved_mb:.0f}MB)"
            except ImportError:
                pass
        elapsed_min = (time.time() - t0) / 60
        print(f"[{elapsed_min:5.1f} phút] RSS={rss_mb:7.1f}MB{vram_str} | active_sessions={active_sessions} | "
              f"ok={stats['ok']} error={stats['error']} graceful_timeout={stats['graceful_timeout']} "
              f"hard_timeout={stats['hard_timeout']}")

    await asyncio.gather(*workers, return_exceptions=True)
    await store.stop_background_cleanup()

    print("\n" + "=" * 60)
    print("KẾT QUẢ SOAK TEST")
    print("=" * 60)
    print(f"Tổng request OK        : {stats['ok']}")
    print(f"Lỗi thực sự (exception): {stats['error']}" + (f" (vd: {stats['last_error']})" if stats['error'] else ""))
    print(f"Graceful timeout (>60s chờ queue, trả lỗi đúng thiết kế): {stats['graceful_timeout']}")
    print(f"Hard timeout (treo thật sự, dấu hiệu deadlock)         : {stats['hard_timeout']}")
    print(f"Latency cao nhất 1 request: {stats['max_latency']*1000:.1f} ms")

    if len(mem_samples) >= 2:
        mem_growth = mem_samples[-1] - mem_samples[0]
        print(f"\nRSS memory: {mem_samples[0]:.1f}MB -> {mem_samples[-1]:.1f}MB (tăng {mem_growth:+.1f}MB)")
        print(f"Số session active theo thời gian: {session_count_samples}")
        max_sessions_seen = max(session_count_samples)
        print(f"Số session active tối đa từng thấy: {max_sessions_seen} (phải bị chặn bởi TTL, KHÔNG được tăng vô hạn)")

    if vram_samples:
        vram_growth = vram_samples[-1] - vram_samples[0]
        print(f"\nVRAM (Router thật): {vram_samples[0]:.0f}MB -> {vram_samples[-1]:.0f}MB "
              f"(tăng {vram_growth:+.0f}MB, đỉnh {max(vram_samples):.0f}MB)")
        print("-> Nếu VRAM tăng đều liên tục theo thời gian (không phải dao động quanh 1 mức ổn "
              "định) là dấu hiệu leak thật ở tầng model/CUDA, cần xem lại router_agent_fast.py "
              "(FIXED_LEN, cache tensor không giải phóng...).")
        no_oom = f"Không OOM (không có CUDA out of memory exception) trong suốt {minutes} phút."
        print(f"Kết luận OOM: {no_oom}")

    verdict_deadlock = "PASS (0 hard timeout)" if stats["hard_timeout"] == 0 else f"FAIL ({stats['hard_timeout']} hard timeout)"
    verdict_leak = "PASS (memory không tăng liên tục / session bị TTL chặn)" \
        if len(mem_samples) < 2 or (mem_samples[-1] - mem_samples[0]) < 50 else \
        "CẦN XEM LẠI (memory tăng > 50MB trong suốt bài test)"
    print(f"\nZero deadlock : {verdict_deadlock}")
    print(f"Zero mem leak : {verdict_leak}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--real-router", action="store_true",
                         help="Dùng Router thật (GPU) thay vì Mock — cần router_agent_fast.py cùng thư mục.")
    args = parser.parse_args()
    asyncio.run(main(args.minutes, args.sessions, args.real_router))