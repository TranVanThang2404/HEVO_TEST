
from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger("request_queue")

T = TypeVar("T")

MAX_CONCURRENT_LLM_REQUESTS = 3     # ngân sách VRAM RTX 3060: giữ nhỏ để Router+Generator không OOM khi chạy song song
QUEUE_WAIT_TIMEOUT_SECONDS = 60.0   # request chờ slot quá lâu -> lỗi graceful (đề bài: > 60s)
DEFAULT_MAX_RETRIES = 3             # tối thiểu 3 lần (đề bài)
DEFAULT_BASE_BACKOFF = 1.5          # giây — nhân đôi mỗi lần retry (exponential)


class RequestTimeoutError(Exception):
    """Request chờ quá lâu trong queue — lỗi graceful, KHÔNG phải crash/deadlock."""


class RequestQueue:
    """
    asyncio.Semaphore trong asyncio hiện đại vốn đã cấp quyền cho các waiter đúng
    thứ tự gọi (FIFO), nên không cần tự cài thêm cấu trúc hàng đợi riêng — chỉ cần
    bọc thêm timeout để đảm bảo "chờ quá 60s -> graceful error" thay vì treo vô thời hạn
    (chính là nguồn deadlock/OOM nếu không kiểm soát).
    """

    def __init__(
        self,
        max_concurrency: int = MAX_CONCURRENT_LLM_REQUESTS,
        wait_timeout: float = QUEUE_WAIT_TIMEOUT_SECONDS,
    ):
        self._sem = asyncio.Semaphore(max_concurrency)
        self._capacity = max_concurrency
        self._wait_timeout = wait_timeout
        self._in_flight = 0
        self._waiting = 0

    async def run(self, coro_fn: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        self._waiting += 1
        try:
            try:
                await asyncio.wait_for(self._sem.acquire(), timeout=self._wait_timeout)
            except asyncio.TimeoutError:
                logger.warning(f"[RequestQueue] Request chờ slot > {self._wait_timeout}s -> trả lỗi graceful.")
                raise RequestTimeoutError(
                    f"Hệ thống đang quá tải, vui lòng thử lại sau (chờ quá {self._wait_timeout:.0f}s)."
                )
        finally:
            self._waiting -= 1

        self._in_flight += 1
        try:
            return await coro_fn(*args, **kwargs)
        finally:
            self._in_flight -= 1
            self._sem.release()

    @property
    def stats(self) -> dict:
        return {"in_flight": self._in_flight, "waiting": self._waiting, "capacity": self._capacity}


def with_retry(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_backoff: float = DEFAULT_BASE_BACKOFF,
    exceptions: tuple = (Exception,),
):
    """
    Decorator retry cho các API call (Router / Generator / Embedding — đề bài liệt kê rõ 3 loại).
    Exponential backoff: delay = base_backoff * (2 ** attempt) + jitter nhỏ để tránh thundering herd.
    """

    def decorator(fn: Callable[..., Awaitable[T]]):
        async def wrapper(*args, **kwargs) -> T:
            last_exc: Exception | None = None
            for attempt in range(max_retries):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt == max_retries - 1:
                        break
                    delay = base_backoff * (2 ** attempt) + random.uniform(0, 0.3)
                    logger.warning(
                        f"[Retry] {fn.__name__} lỗi lần {attempt + 1}/{max_retries} ({e}) "
                        f"-> thử lại sau {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
            logger.error(f"[Retry] {fn.__name__} thất bại sau {max_retries} lần thử: {last_exc}")
            raise last_exc

        return wrapper

    return decorator