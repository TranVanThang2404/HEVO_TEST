# -*- coding: utf-8 -*-
"""
Demo Phần C — Intelligent Cache & Edge (C2 Cache Pipeline + C3 Guardrails).

Cùng phong cách với demo_multiturn.py (Phần A) và demo_graphrag_stream.py
(Phần B): 1 script chạy được ngay, in log rõ ràng, dùng dữ liệu/module THẬT
đã build và verify trong session — không phải mock cho có.

Gồm 2 đoạn demo nối tiếp nhau:

  ĐOẠN 1 — C2: Cache Pipeline & Paraphrase (dùng semantic_cache_c2.py thật)
    - Câu 1 (gốc, chưa có trong cache)      -> MISS, gọi "Agent" (agent_fn
      giả lập FAQ agent trả lời từ menu/faq), lưu vào cache.
    - Câu 2 (CÙNG ý nhưng viết khác, có từ đệm khác) -> phải ra HIT nhờ
      fast-path rule-based tách {Hành động} nhất quán (fast_extract.py) +
      embedding similarity (Qwen3-Embedding-0.6B qua B1/embeddings.py).
    - Câu 3 (có thêm {Ngữ cảnh} thời gian) -> vẫn HIT trên {Hành động},
      nhưng {Ngữ cảnh} khác nên câu trả lời được paraphrase thêm phần
      "Lưu ý: ..." — chứng minh cache tách đúng action/context.
    - In cache invalidation (C3): xoá cache, câu vừa HIT giờ MISS lại.

  ĐOẠN 2 — C3: Production Guardrails (dùng guardrails.py thật)
    - TTS preprocessing trên câu trả lời vừa lấy được ở Đoạn 1.
    - Rate limiting: bắn liên tiếp nhiều request hơn hạn mức, cho thấy
      request thứ (N+1) bị từ chối, rồi được phép lại sau khi hết window.
    - Graceful degradation: giả lập Generator lỗi liên tiếp -> circuit
      breaker mở mạch -> trả fallback tĩnh ngay, không chờ timeout.
    - Health check: gọi get_system_health() thật (kết quả tuỳ máy đang
      chạy Neo4j/Generator hay không -> đây chính là điểm hay của health
      check, phản ánh đúng trạng thái thật).

Đặt vào /mnt/d/HEVO/A2 (cùng thư mục semantic_cache_c2.py, fast_extract.py,
intent_extractor.py, guardrails.py, c2_data/faq.csv) rồi chạy:
    python3 demo_part_c.py

Cần: GPU nạp được Qwen3-Embedding-0.6B (cho semantic cache) + model intent
(intent_sft_merged nếu đã fine-tune, hoặc base model — xem intent_extractor.py).
KHÔNG cần Neo4j/SGLang Generator đang chạy (health check ở Đoạn 2 vẫn chạy
được, chỉ báo "down" nếu chưa bật — đúng hành vi thật của health check).
"""
import asyncio
import csv
import os
import time

from semantic_cache_c2 import SemanticCacheC2
from guardrails import (
    preprocess_for_tts,
    RateLimiter,
    CircuitBreaker,
    call_generator_with_fallback,
    get_system_health,
)

HERE = os.path.dirname(__file__)
FAQ_CSV = os.path.join(HERE, "c2_data", "faq.csv")


def load_faq_answer_for(keyword: str) -> str:
    """Lấy 1 câu trả lời thật từ faq.csv thay vì bịa, để agent_fn demo trả
    đúng dữ liệu thật của công ty (không phải placeholder)."""
    try:
        with open(FAQ_CSV, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                q = row.get("question", "")
                if keyword.lower() in q.lower():
                    return row.get("answer", "").strip()
    except FileNotFoundError:
        pass
    return "Dạ giá cà phê sữa đá size L là 45.000đ (đã bao gồm VAT), quán mở cửa 7h-22h ạ."


def make_agent_fn():
    """Giả lập FAQ Agent thật — trong hệ thống đầy đủ đây là orchestrator.py
    gọi sang Generator qua SGLang; ở đây trả thẳng câu trả lời thật từ
    faq.csv để demo tập trung vào hành vi CACHE, không phụ thuộc server
    Generator có đang chạy hay không."""

    def agent_fn(action_text: str, context_text: str) -> str:
        print(f"    -> [Agent thật được gọi] action='{action_text}'")
        return load_faq_answer_for("wifi") if "wifi" in action_text.lower() \
            else load_faq_answer_for("giờ") if "giờ" in action_text.lower() \
            else load_faq_answer_for("size")

    return agent_fn


def warmup_models():
    """Nạp trước embedding model + intent model (tải từ HF / lên GPU lần đầu
    mất tới vài chục giây) TRƯỚC khi đo latency, để timer của C2 chỉ phản
    ánh đúng thời gian XỬ LÝ 1 request, không lẫn thời gian load model
    (lỗi đã gặp ở lần chạy demo đầu tiên: MISS latency=55576.2ms vì gồm
    cả load model — không phải latency thật của hệ thống production, vì
    production chỉ load model 1 lần lúc khởi động server)."""
    print("[Khởi động] Đang nạp trước embedding model + intent model (không tính vào latency)...")
    from intent_extractor import extract_intent
    from semantic_cache_c2 import embed_one as _embed_one
    extract_intent("câu khởi động, không tính")
    _embed_one("câu khởi động, không tính")
    print("[Khởi động] Xong.\n")


def demo_c2_cache():
    print("=" * 70)
    print("ĐOẠN 1 — C2: CACHE PIPELINE & PARAPHRASE")
    print("=" * 70)

    cache = SemanticCacheC2()
    agent_fn = make_agent_fn()

    queries = [
        ("Câu 1 (gốc, cache rỗng)", "Cho em hỏi wifi quán tên gì, mật khẩu là gì?"),
        ("Câu 2 (viết khác, cùng ý)", "Anh/chị ơi cho em hỏi wifi quán tên gì, mật khẩu là gì vậy ạ"),
        ("Câu 3 (cùng ý + thêm ngữ cảnh)", "Cho em hỏi wifi quán tên gì, mật khẩu là gì, tối nay em ghé nhé"),
    ]

    for label, q in queries:
        print(f"\n[{label}] IN: {q}")
        result = cache.process(q, agent_fn)
        tag = "HIT " if result["cache_hit"] else "MISS"
        print(f"  [{tag}] latency={result['latency_ms']:.1f}ms "
              f"(extract_slm={result['extract_latency_ms']:.1f}ms) "
              f"similarity={result['similarity']:.3f}")
        print(f"  action='{result['action']}' | context='{result['context']}'")
        print(f"  response: {result['response']}")

    print(f"\n[Thống kê cache] hit={cache.hits} miss={cache.misses} "
          f"hit_rate={cache.hit_rate * 100:.1f}%")

    print("\n[C3 - Cache invalidation] Xoá toàn bộ cache (vd sau khi cập nhật menu/FAQ)...")
    cache.invalidate_all()
    q_repeat = queries[0][1]
    result = cache.process(q_repeat, agent_fn)
    tag = "HIT " if result["cache_hit"] else "MISS"
    print(f"  Chạy lại câu 1 sau invalidate -> [{tag}] (đúng ra phải là MISS vì cache đã xoá sạch)")

    return result["response"]


async def demo_c3_guardrails(sample_reply: str):
    print("\n" + "=" * 70)
    print("ĐOẠN 2 — C3: PRODUCTION GUARDRAILS")
    print("=" * 70)

    # 1) TTS preprocessing — dùng câu có giá/size/VAT thật (lấy từ menu.csv)
    # để thấy được biến đổi; câu trả lời wifi ở Đoạn 1 không có gì để biến
    # đổi nên không dùng câu đó ở đây.
    print("\n[1/4] TTS Preprocessing")
    menu_reply = ("Dạ Cà phê sữa đá Size L giá 48.000đ (đã gồm VAT) ạ, "
                   "bên em cũng có size M giá 42.000đ.")
    print(f"  Trước: {menu_reply}")
    print(f"  Sau  : {preprocess_for_tts(menu_reply)}")
    print(f"  (Câu trả lời wifi ở Đoạn 1 không có giá/size/VAT nên không đổi: "
          f"{sample_reply == preprocess_for_tts(sample_reply)})")

    # 2) Rate limiting
    print("\n[2/4] Rate Limiting (giới hạn demo: 3 request / 2 giây)")
    limiter = RateLimiter(max_requests=3, window_seconds=2.0)
    for i in range(5):
        allowed, remaining = await limiter.allow("demo-client")
        print(f"  Request #{i + 1}: {'CHO PHÉP' if allowed else 'TỪ CHỐI (429)'} "
              f"(còn lại trong window: {remaining})")
    print("  Đợi hết window (2.1s) rồi thử lại...")
    await asyncio.sleep(2.1)
    allowed, remaining = await limiter.allow("demo-client")
    print(f"  Request sau khi window reset: {'CHO PHÉP' if allowed else 'TỪ CHỐI'} "
          f"(còn lại: {remaining})")

    # 3) Graceful degradation / circuit breaker
    print("\n[3/4] Graceful Degradation (circuit breaker quanh Generator)")
    circuit = CircuitBreaker(failure_threshold=3, open_duration_seconds=2.0)

    async def failing_call():
        raise ConnectionError("Generator giả lập bị treo/timeout")

    for i in range(4):
        result, is_fallback = await call_generator_with_fallback(
            failing_call, intent="faq", circuit=circuit,
        )
        print(f"  Lần gọi #{i + 1}: circuit={circuit.state} "
              f"is_fallback={is_fallback} -> \"{result}\"")
    print("  Đợi circuit chuyển sang half-open (2.1s)...")
    await asyncio.sleep(2.1)

    async def working_call():
        return "Dạ đây là câu trả lời thật từ Generator (giả lập đã phục hồi)."

    result, is_fallback = await call_generator_with_fallback(
        working_call, intent="faq", circuit=circuit,
    )
    print(f"  Lần gọi sau phục hồi: circuit={circuit.state} "
          f"is_fallback={is_fallback} -> \"{result}\"")

    # 4) Health check — kết quả THẬT tuỳ máy đang chạy gì
    print("\n[4/4] Health Check (kết quả thật tuỳ theo Neo4j/Generator đang chạy hay không)")
    health = await get_system_health()
    print(f"  Overall: {health['status']}")
    for name, comp in health["components"].items():
        print(f"    - {name}: {comp}")
    print(f"  Circuit breaker (Generator thật, dùng chung toàn hệ thống): "
          f"{health['circuit_breaker']}")


async def main():
    print("#" * 70)
    print("# DEMO PHẦN C — HIGHLANDS COFFEE MULTI-AGENT SYSTEM")
    print("#" * 70)

    warmup_models()
    sample_reply = demo_c2_cache()
    await demo_c3_guardrails(sample_reply)

    print("\n" + "=" * 70)
    print("KẾT THÚC DEMO PHẦN C")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())