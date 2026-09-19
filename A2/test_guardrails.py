# -*- coding: utf-8 -*-
"""
Test độc lập cho guardrails.py (Phần C3).

Chạy trên laptop, KHÔNG cần GPU. Chỉ cần Neo4j đang chạy nếu muốn test
mục 4 (health check) đầy đủ — nếu Neo4j tắt thì health check Neo4j sẽ
báo "down" (đúng hành vi mong đợi, không phải lỗi script).

Đặt vào /mnt/d/HEVO/A2 (cùng thư mục guardrails.py) rồi chạy:
    python3 test_guardrails.py
"""
import asyncio
from guardrails import (
    preprocess_for_tts,
    RateLimiter,
    CircuitBreaker,
    call_generator_with_fallback,
    get_system_health,
)


def test_tts_preprocessing():
    print("=" * 70)
    print("[1/4] TTS PREPROCESSING")
    print("=" * 70)
    cases = [
        "Cà phê sữa đá size L giá 49.000đ ạ.",
        "Trà đào size M là 55000đ, chưa gồm VAT.",
        "Bạc xỉu size S chỉ 39,000đ thôi ạ.",
        "Anh muốn thêm size XL không ạ, giá 65.000đ.",
        "Không có giá tiền hay size trong câu này.",
    ]
    for c in cases:
        out = preprocess_for_tts(c)
        mark = "✅" if out != c else "⚪"
        print(f"{mark} IN : {c}")
        print(f"   OUT: {out}\n")


async def test_rate_limiter():
    print("=" * 70)
    print("[2/4] RATE LIMITING (giới hạn 3 request / 2 giây cho test nhanh)")
    print("=" * 70)
    limiter = RateLimiter(max_requests=3, window_seconds=2.0)
    client = "demo-client-ip"
    for i in range(1, 6):
        allowed, remaining = await limiter.allow(client)
        status = "✅ CHO PHÉP" if allowed else "❌ TỪ CHỐI (429)"
        print(f"  Request #{i}: {status}  (còn lại trong window: {remaining})")
    print("  -> Chờ 2.1s để window reset...")
    await asyncio.sleep(2.1)
    allowed, remaining = await limiter.allow(client)
    print(f"  Request #6 (sau khi window reset): {'✅ CHO PHÉP' if allowed else '❌ TỪ CHỐI'}")
    print()


async def test_circuit_breaker():
    print("=" * 70)
    print("[3/4] GRACEFUL DEGRADATION (circuit breaker quanh Generator)")
    print("=" * 70)
    breaker = CircuitBreaker(failure_threshold=3, open_duration_seconds=3.0)

    async def _always_fail():
        raise ConnectionError("Giả lập Generator quá tải / không phản hồi")

    async def _always_succeed():
        return "Đây là câu trả lời thật từ Generator."

    print("  Mô phỏng Generator LIÊN TỤC LỖI (3 lần đầu để mở mạch):")
    for i in range(1, 5):
        result, is_fallback = await call_generator_with_fallback(
            _always_fail, intent="order", circuit=breaker
        )
        tag = "FALLBACK" if is_fallback else "THẬT"
        print(f"    Lần {i}: [{tag}] circuit={breaker.state}  -> {result}")

    print(f"\n  Circuit hiện tại: {breaker.state} (nên là OPEN)")
    print("  Chờ 3.1s để circuit tự thử lại (half-open)...")
    await asyncio.sleep(3.1)

    print("  Generator ĐÃ PHỤC HỒI, gọi lại:")
    result, is_fallback = await call_generator_with_fallback(
        _always_succeed, intent="order", circuit=breaker
    )
    tag = "FALLBACK" if is_fallback else "THẬT"
    print(f"    -> [{tag}] circuit={breaker.state}  -> {result}")
    print()


async def test_health_check():
    print("=" * 70)
    print("[4/4] HEALTH CHECK (cần Generator ở port 30001 + Neo4j để test đầy đủ)")
    print("=" * 70)
    health = await get_system_health(router_client=None)
    import json
    print(json.dumps(health, ensure_ascii=False, indent=2))
    print()


async def main():
    test_tts_preprocessing()
    await test_rate_limiter()
    await test_circuit_breaker()
    await test_health_check()
    print("=" * 70)
    print("HOÀN TẤT TEST GUARDRAILS (C3)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())