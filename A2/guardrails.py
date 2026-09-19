# -*- coding: utf-8 -*-
"""
Phần C3 — Production Guardrails (3%).

Module độc lập, KHÔNG phụ thuộc GPU/model nặng, chỉ cần httpx (đã có sẵn
trong project vì api_server.py/demo_graphrag_stream.py đã dùng).

Bao gồm đúng 4 yêu cầu của đề bài (trang 13):
  1. TTS Preprocessing   -> preprocess_for_tts()
  2. Rate limiting        -> RateLimiter (+ FastAPI dependency rate_limit_dependency)
  3. Graceful degradation -> CircuitBreaker + call_generator_with_fallback()
  4. Health check         -> get_system_health()

Đặt file này vào /mnt/d/HEVO/A2 (cùng thư mục api_server.py), rồi xem
GHI CHÚ TÍCH HỢP ở cuối file để nối vào api_server.py thật.
"""
from __future__ import annotations

import re
import time
import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, Any

import httpx


# ============================================================
# 1. TTS PREPROCESSING
# ============================================================
# Chuẩn hoá text trước khi đưa qua TTS: giá tiền, size, VAT.
# Giả định (ghi rõ để báo cáo trung thực): đề bài cho ví dụ
# "49.000đ" -> "49k" nghĩa là rút gọn số cho TTS đọc nhanh/tự nhiên
# hơn là đọc đầy đủ "bốn mươi chín nghìn đồng". Size S/M/L/XL được
# đọc ra tiếng Việt tự nhiên (nhỏ/vừa/lớn/siêu lớn) vì TTS tiếng Việt
# đọc chữ cái rời rạc (S, M, L) rất khó nghe. VAT -> "thuế".

_CURRENCY_RE = re.compile(
    r"(?<!\w)(\d{1,3}(?:[.,]\d{3})+|\d{4,})\s*(?:đ|d|vnđ|vnd)\b",
    re.IGNORECASE,
)

_SIZE_MAP = {
    "S": "nhỏ",
    "M": "vừa",
    "L": "lớn",
    "XL": "siêu lớn",
}
_SIZE_RE = re.compile(
    r"\bsize\s+(S|M|L|XL)\b",
    re.IGNORECASE,
)

_VAT_RE = re.compile(r"\bVAT\b", re.IGNORECASE)


def _currency_repl(m: "re.Match") -> str:
    raw = m.group(1)
    digits = re.sub(r"[.,]", "", raw)
    try:
        value = int(digits)
    except ValueError:
        return m.group(0)
    if value <= 0:
        return m.group(0)
    if value % 1000 == 0:
        return f"{value // 1000}k"
    return f"{value / 1000:.1f}k"


def _size_repl(m: "re.Match") -> str:
    key = m.group(1).upper()
    return f"size {_SIZE_MAP.get(key, key)}"


def preprocess_for_tts(text: str) -> str:
    """Chuẩn hoá 1 câu trả lời của Generator trước khi phát TTS.

    - "49.000đ" / "49000đ" / "49,000đ" -> "49k"
    - "size M" -> "size vừa" (S/M/L/XL)
    - "VAT" -> "thuế"
    """
    if not text:
        return text
    out = _CURRENCY_RE.sub(_currency_repl, text)
    out = _SIZE_RE.sub(_size_repl, out)
    out = _VAT_RE.sub("thuế", out)
    return out


# ============================================================
# 2. RATE LIMITING
# ============================================================
# Sliding-window counter đơn giản, in-memory, theo client key
# (vd IP hoặc session_id). Không cần Redis vì đây là demo 1 máy.


class RateLimiter:
    def __init__(self, max_requests: int = 10, window_seconds: float = 10.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = {}
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> tuple[bool, int]:
        """Trả về (được_phép, số_request_còn_lại_trong_window)."""
        now = time.monotonic()
        async with self._lock:
            dq = self._hits.setdefault(key, deque())
            cutoff = now - self.window_seconds
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.max_requests:
                return False, 0
            dq.append(now)
            return True, self.max_requests - len(dq)


# Instance dùng chung cho toàn bộ API (10 request / 10 giây / client)
GLOBAL_RATE_LIMITER = RateLimiter(max_requests=10, window_seconds=10.0)


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: float = 10.0):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded, retry after {retry_after}s")


async def enforce_rate_limit(client_key: str, limiter: RateLimiter = GLOBAL_RATE_LIMITER) -> None:
    """Gọi ở đầu mỗi endpoint (hoặc làm FastAPI Depends). Raise nếu vượt hạn mức."""
    allowed, _remaining = await limiter.allow(client_key)
    if not allowed:
        raise RateLimitExceeded(retry_after=limiter.window_seconds)


# ============================================================
# 3. GRACEFUL DEGRADATION — Circuit breaker quanh SGLang Generator
# ============================================================
# Nếu Generator lỗi/timeout liên tiếp -> "mở mạch" (open circuit) trong
# một khoảng thời gian, trả lời fallback tĩnh ngay lập tức thay vì
# chờ timeout lặp lại nhiều lần (mỗi lần user hỏi lại là 1 lần treo app).

FALLBACK_TEMPLATES = {
    "order": "Dạ hiện hệ thống đang tải cao, anh/chị vui lòng đợi nhân viên xác nhận đơn giúp em nhé ạ.",
    "consultant": "Dạ hiện em chưa tư vấn được ngay do hệ thống đang bận, anh/chị đợi chút hoặc ghé quầy để được tư vấn trực tiếp nhé ạ.",
    "faq": "Dạ hệ thống đang bận, anh/chị vui lòng liên hệ trực tiếp nhân viên quầy để được hỗ trợ nhanh nhất ạ.",
    "ignore": "Dạ em nghe chưa rõ, anh/chị nói lại giúp em nhé.",
}
DEFAULT_FALLBACK = "Dạ hệ thống đang gặp sự cố tạm thời, anh/chị vui lòng thử lại sau ít phút hoặc liên hệ nhân viên quầy giúp em ạ."


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3          # số lỗi liên tiếp để mở mạch
    open_duration_seconds: float = 30.0  # thời gian giữ mạch mở
    _consecutive_failures: int = field(default=0, init=False)
    _opened_at: Optional[float] = field(default=None, init=False)

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.open_duration_seconds:
            # hết thời gian mở mạch -> cho thử lại (half-open)
            self._opened_at = None
            self._consecutive_failures = 0
            return False
        return True

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold and self._opened_at is None:
            self._opened_at = time.monotonic()

    @property
    def state(self) -> str:
        return "OPEN" if self.is_open() else "CLOSED"


GENERATOR_CIRCUIT = CircuitBreaker(failure_threshold=3, open_duration_seconds=30.0)


async def call_generator_with_fallback(
    call_fn: Callable[[], Awaitable[Any]],
    intent: str = "faq",
    circuit: CircuitBreaker = GENERATOR_CIRCUIT,
    timeout_seconds: float = 10.0,
) -> tuple[Any, bool]:
    """Bọc quanh lệnh gọi Generator thật (call_fn).

    Trả về (kết_quả_hoặc_None, is_fallback: bool).
    - Nếu circuit đang OPEN -> trả fallback ngay, KHÔNG gọi Generator (tránh
      dồn thêm tải/timeout lên 1 service đang quá tải).
    - Nếu gọi thật bị lỗi/timeout -> ghi nhận failure, trả fallback.
    - Nếu gọi thật thành công -> ghi nhận success, trả kết quả thật.
    """
    if circuit.is_open():
        return FALLBACK_TEMPLATES.get(intent, DEFAULT_FALLBACK), True

    try:
        result = await asyncio.wait_for(call_fn(), timeout=timeout_seconds)
        circuit.record_success()
        return result, False
    except (httpx.HTTPError, asyncio.TimeoutError, ConnectionError) as exc:
        circuit.record_failure()
        return FALLBACK_TEMPLATES.get(intent, DEFAULT_FALLBACK), True


# ============================================================
# 4. HEALTH CHECK ENDPOINTS
# ============================================================
# Kiểm tra từng service con: Generator (SGLang), Neo4j, Router (đã nạp
# lên GPU chưa). Dùng cho GET /health/detailed trong api_server.py.

SGLANG_GENERATOR_URL_BASE = "http://127.0.0.1:30001"


async def check_generator_health(base_url: str = SGLANG_GENERATOR_URL_BASE, timeout: float = 2.0) -> dict:
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{base_url}/v1/models")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            return {"status": "up", "latency_ms": round(elapsed_ms, 1)}
        return {"status": "degraded", "http_status": resp.status_code, "latency_ms": round(elapsed_ms, 1)}
    except Exception as exc:  # noqa: BLE001 — health check phải bắt mọi lỗi, không được crash
        return {"status": "down", "error": str(exc)}


async def check_neo4j_health(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    timeout_seconds: float = 3.0,
) -> dict:
    t0 = time.perf_counter()
    try:
        # import trễ để module này không bắt buộc cài neo4j driver nếu chỉ test TTS/rate-limit
        from neo4j import GraphDatabase  # type: ignore

        import sys, os
        _b1_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "B1"))
        if _b1_dir not in sys.path:
            sys.path.insert(0, _b1_dir)
        import hybrid_search as _hs  # type: ignore

        uri = uri or _hs.NEO4J_URI
        user = user or _hs.NEO4J_USER
        password = password or _hs.NEO4J_PASSWORD

        def _ping():
            # connection_timeout ngăn driver treo lâu (mặc định có thể rất lâu)
            # khi Neo4j không chạy hoặc không reachable.
            driver = GraphDatabase.driver(
                uri, auth=(user, password), connection_timeout=timeout_seconds
            )
            try:
                with driver.session() as session:
                    session.run("RETURN 1").consume()
            finally:
                driver.close()

        await asyncio.wait_for(asyncio.to_thread(_ping), timeout=timeout_seconds + 1.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {"status": "up", "latency_ms": round(elapsed_ms, 1)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "down", "error": str(exc)}


def check_router_health(router_client: Optional[object] = None) -> dict:
    """Router chạy in-process (không qua HTTP) nên health check là kiểm tra
    xem model đã được nạp lên GPU trong process hiện tại hay chưa."""
    if router_client is None:
        return {"status": "unknown", "detail": "router_client không được truyền vào"}
    loaded = getattr(router_client, "_classify_intent", None) is not None
    return {"status": "up" if loaded else "down"}


async def get_system_health(router_client: Optional[object] = None) -> dict:
    generator, neo4j = await asyncio.gather(
        check_generator_health(),
        check_neo4j_health(),
    )
    router = check_router_health(router_client)
    components = {"generator": generator, "neo4j": neo4j, "router": router}
    overall = "up" if all(c.get("status") == "up" for c in components.values()) else "degraded"
    if all(c.get("status") == "down" for c in components.values()):
        overall = "down"
    return {"status": overall, "components": components, "circuit_breaker": GENERATOR_CIRCUIT.state}


# ============================================================
# GHI CHÚ TÍCH HỢP VÀO api_server.py THẬT (không tự động sửa file):
#
# 1) TTS preprocessing — trong chỗ trả response cuối cùng cho client
#    (sau khi Generator sinh xong câu trả lời, TRƯỚC khi gửi cho TTS):
#       from guardrails import preprocess_for_tts
#       final_text = preprocess_for_tts(full_reply)
#
# 2) Rate limiting — thêm vào đầu mỗi route (vd /v1/chat/stream):
#       from guardrails import enforce_rate_limit, RateLimitExceeded
#       from fastapi import Request, HTTPException
#       @app.post("/v1/chat/stream")
#       async def chat_stream(request: Request, ...):
#           client_key = request.client.host if request.client else "unknown"
#           try:
#               await enforce_rate_limit(client_key)
#           except RateLimitExceeded as e:
#               raise HTTPException(status_code=429, detail="Too many requests",
#                                    headers={"Retry-After": str(int(e.retry_after))})
#
# 3) Graceful degradation — bọc quanh lệnh gọi SGLang trong stream_from_sglang():
#       from guardrails import call_generator_with_fallback
#       result, is_fallback = await call_generator_with_fallback(
#           call_fn=lambda: _real_call_sglang(...), intent=detected_intent)
#       # nếu is_fallback=True thì result là câu fallback tĩnh, phát luôn
#       # không cần gọi SGLang nữa (tránh app bị treo khi Generator quá tải)
#
# 4) Health check — thêm route mới (giữ nguyên /health cũ để không phá API hiện có):
#       from guardrails import get_system_health
#       @app.get("/health/detailed")
#       async def health_detailed():
#           return await get_system_health(router_client=router)  # router = LocalSLMRouterClient instance
# ============================================================