# -*- coding: utf-8 -*-
"""
Phần C2.2 — Cache Pipeline & Paraphrase.

Luồng đúng theo đề bài (trang 13):
  1. Extract: intent_extractor.extract_intent() tách {subject}/{action}/{context}.
  2. Query Cache: embedding similarity (>= 0.92) trên phần {action} để tìm
     trong Cache DB (ở đây là in-memory, có thể thay bằng Redis/vector DB
     thật khi lên production).
  3. Cache Hit -> Paraphrase trực tiếp: lấy response template đã lưu, kết
     hợp với {context} -> câu trả lời hoàn chỉnh. Xử lý tại RAM, KHÔNG gọi Agent.
  4. Cache Miss -> gọi Agent thật (agent_fn được truyền vào), lưu kết quả
     vào cache để lần sau hit.

Dùng lại real_embed_fn cùng không gian vector với GraphRAG (Qwen3-Embedding-0.6B
qua B1/embeddings.py.embed_one()) để nhất quán với cache_layer.py đã có.

Đặt vào /mnt/d/HEVO/A2 (cùng thư mục intent_extractor.py, cache_layer.py).
"""
import os
import sys
import time

import numpy as np

from intent_extractor import extract_intent
from fast_extract import strip_wrappers, extract_context_fast, guess_subject_fast

_B1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "B1"))
if _B1_DIR not in sys.path:
    sys.path.insert(0, _B1_DIR)
from embeddings import embed_one  # type: ignore  # noqa: E402

SIMILARITY_THRESHOLD = 0.92


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SemanticCacheC2:
    """Cache theo phần {Hành động}: key = embedding(action), value = response template."""

    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold
        self._entries: list[dict] = []
        self.hits = 0
        self.misses = 0

    def _find_match(self, action_embedding):
        best_score, best_entry = 0.0, None
        for entry in self._entries:
            score = _cosine(action_embedding, entry["embedding"])
            if score > best_score:
                best_score, best_entry = score, entry
        if best_entry is not None and best_score >= self.similarity_threshold:
            return best_entry, best_score
        return None, best_score

    def invalidate_all(self) -> None:
        """C3 — cache invalidation khi menu/FAQ thay đổi: xoá sạch cache."""
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def process(self, user_query: str, agent_fn):
        """agent_fn(action_text, context_text) -> response_template thật.

        Trả về dict: response, latency_ms, cache_hit, action, context, similarity.

        KIẾN TRÚC (xem ghi chú trong fast_extract.py): {Hành động} dùng làm
        cache-key luôn được tách bằng rule-based fast_extract (regex, <1ms),
        chạy cho MỌI request kể cả khi cache hit -> giữ latency nhánh hit
        thấp và nhất quán giữa các biến thể cùng câu hỏi. SLM Intent
        Extraction (intent_extractor.py) CHỈ chạy khi cache MISS thật sự,
        để lấy {Chủ ngữ} chính xác hơn cho response — không nằm trên
        đường găng (critical path) của nhánh hit.
        """
        t0 = time.perf_counter()
        action_text = strip_wrappers(user_query)
        context_text = extract_context_fast(user_query)

        action_embedding = embed_one(action_text)
        match, score = self._find_match(action_embedding)

        if match is not None:
            response = self._paraphrase(match["response_template"], context_text)
            self.hits += 1
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "response": response,
                "latency_ms": latency_ms,
                "extract_latency_ms": 0.0,  # fast-path, không gọi SLM
                "cache_hit": True,
                "action": action_text,
                "context": context_text,
                "similarity": score,
            }

        # Cache MISS -> gọi SLM Intent Extraction đầy đủ (C2.1) để lấy
        # {Chủ ngữ} chính xác hơn, rồi gọi Agent thật xử lý.
        parsed, extract_latency_ms, _raw = extract_intent(user_query)
        subject = parsed.get("subject") or guess_subject_fast(user_query)

        response_template = agent_fn(action_text, context_text)
        self._entries.append(
            {
                "action_text": action_text,
                "embedding": action_embedding,
                "response_template": response_template,
                "subject": subject,
            }
        )
        self.misses += 1
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "response": self._paraphrase(response_template, context_text),
            "latency_ms": latency_ms,
            "extract_latency_ms": extract_latency_ms,
            "cache_hit": False,
            "action": action_text,
            "context": context_text,
            "similarity": score,
            "subject": subject,
        }

    @staticmethod
    def _paraphrase(template: str, context_text: str) -> str:
        if context_text:
            return f"{template} (Lưu ý: {context_text})"
        return template

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0