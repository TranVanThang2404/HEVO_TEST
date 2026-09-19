
from __future__ import annotations

import asyncio
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

logger = logging.getLogger("session_store")

HISTORY_WINDOW = 5              # tối thiểu 5 câu hỏi gần nhất (đề bài)
CONTEXT_WINDOW_TOKENS = 2048    # context window giả định của Generator agent
SUMMARIZE_THRESHOLD_RATIO = 0.7  # ~70% context window (đề bài)
SESSION_TTL_SECONDS = 30 * 60    # 30 phút (đề bài)
CLEANUP_INTERVAL_SECONDS = 60    # quét dọn định kỳ mỗi 60s


def _rough_token_count(text: str) -> int:
    """Ước lượng nhanh số token mà không cần load tokenizer thật ở tầng session.
    Kinh nghiệm: văn bản Việt/Anh trộn ~ 1 token / 2.5-3 ký tự."""
    return max(1, len(text) // 3)


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    text: str
    ts: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_WINDOW * 2))
    summary: str = ""
    last_active: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def touch(self):
        self.last_active = time.time()

    def total_tokens(self) -> int:
        return _rough_token_count(self.summary) + sum(_rough_token_count(t.text) for t in self.history)


SummarizeFn = Callable[[str, list], Awaitable[str]]


class SessionStore:
    def __init__(self, summarize_fn: Optional[SummarizeFn] = None):
        self._sessions: dict[str, Session] = {}
        self._summarize_fn = summarize_fn  # async (old_summary, turns) -> new_summary_text
        self._cleanup_task: Optional[asyncio.Task] = None

    def get_or_create(self, session_id: str) -> Session:
        sess = self._sessions.get(session_id)
        if sess is None:
            sess = Session(session_id=session_id)
            self._sessions[session_id] = sess
            logger.info(f"[SessionStore] Tạo session mới: {session_id}")
        return sess

    async def add_turn(self, session_id: str, role: str, text: str):
        sess = self.get_or_create(session_id)
        async with sess.lock:
            sess.history.append(Turn(role=role, text=text))
            sess.touch()
            threshold = CONTEXT_WINDOW_TOKENS * SUMMARIZE_THRESHOLD_RATIO
            if sess.total_tokens() > threshold and self._summarize_fn is not None:
                await self._auto_summarize(sess)

    async def _auto_summarize(self, sess: Session):
        # Giữ lại HISTORY_WINDOW turn gần nhất nguyên văn, phần cũ hơn đem tóm tắt.
        keep = list(sess.history)[-HISTORY_WINDOW:]
        to_summarize = list(sess.history)[:-HISTORY_WINDOW]
        if not to_summarize:
            return
        logger.info(
            f"[SessionStore] Session {sess.session_id} vượt {SUMMARIZE_THRESHOLD_RATIO * 100:.0f}% "
            f"context ({sess.total_tokens()} token ước tính) -> auto-summarize {len(to_summarize)} turn cũ."
        )
        try:
            new_summary = await self._summarize_fn(sess.summary, to_summarize)
            sess.summary = new_summary
            sess.history = deque(keep, maxlen=HISTORY_WINDOW * 2)
        except Exception as e:
            logger.warning(f"[SessionStore] Auto-summarize thất bại ({e}), giữ nguyên lịch sử.")

    def get_context(self, session_id: str) -> tuple[str, list]:
        """Trả về (summary, list các turn gần nhất) để agent build prompt."""
        sess = self._sessions.get(session_id)
        if sess is None:
            return "", []
        return sess.summary, list(sess.history)

    async def start_background_cleanup(self):
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("[SessionStore] Background cleanup task đã khởi động.")

    async def stop_background_cleanup(self):
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
            self.cleanup_expired()

    def cleanup_expired(self):
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if now - s.last_active > SESSION_TTL_SECONDS]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info(f"[SessionStore] Đã dọn {len(expired)} session hết hạn (TTL {SESSION_TTL_SECONDS}s): {expired}")

    def active_session_count(self) -> int:
        return len(self._sessions)