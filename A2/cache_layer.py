"""
B2.3 — Kiến trúc cache đa tầng cho hệ thống Multi-Agent + Graph RAG.

    Tầng cache      | Vai trò                                         | Nơi thực hiện
    ---------------------------------------------------------------------------------------
    Model Cache     | Không load lại trọng số model mỗi request       | ModelCache (module này)
    Graph Cache     | Cache kết quả truy vấn Graph RAG (B1, Neo4j)    | GraphCache (module này)
    KV Cache        | Tái sử dụng KV-cache của LLM khi prefix trùng   | CÓ SẴN trong SGLang (radix-tree
                    | (system prompt, few-shot, hội thoại đã sinh)    | prefix cache) — bật qua flag
                    |                                                  | --schedule-policy lpm khi launch
                    |                                                  | server, không tự viết lại ở app-layer.
                    |                                                  | Verify bằng benchmark_kv_cache.py.
    Session Cache   | Lưu lịch sử hội thoại theo session_id           | SessionStore (đã làm ở A2.2,
                    |                                                  | session_store.py) — TÁI SỬ DỤNG,
                    |                                                  | không viết lại.
    ---------------------------------------------------------------------------------------
    Semantic Cache  | Cache theo ĐỘ TƯƠNG ĐỒNG NGỮ NGHĨA (bonus,      | SemanticCache (module này)
    (bonus)         | không cần khớp y hệt chuỗi, threshold >= 0.95)  |

Model Cache và Graph Cache là app-level cache tự viết (in-process, dùng dict + TTL +
LRU eviction, không cần Redis để đơn giản hoá — có thể swap sang Redis sau nếu cần
scale nhiều worker process).

Semantic Cache dùng embedding pluggable (embed_fn): mặc định có 1 embedding rất nhẹ
(TF-IDF/bag-of-words cosine) để chạy demo được ngay không cần model ngoài; khi tích
hợp thật, truyền vào embed_fn = embedding model đã dùng cho Graph RAG hybrid_search
(B1) để đảm bảo nhất quán không gian vector.

Chạy demo:
    python3 cache_layer.py
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ============================================================================
# 1. MODEL CACHE — không load lại model mỗi request
# ============================================================================

class ModelCache:
    """Singleton registry giữ model đã load trong RAM/VRAM, tránh load lại tốn
    thời gian + VRAM mỗi khi có request. Trong kiến trúc SGLang hiện tại, việc
    này vốn đã được đảm bảo vì server chạy long-lived (model load 1 lần lúc
    khởi động). ModelCache ở đây phục vụ các thành phần load model TRỰC TIẾP
    trong process Python (ví dụ embedding model của Graph RAG, hoặc khi test
    AutoAWQ độc lập không qua SGLang) để không vô tình load lại nhiều lần.
    """

    def __init__(self):
        self._models: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0

    def get_or_load(self, key: str, loader_fn: Callable[[], Any]) -> Any:
        if key in self._models:
            self.hits += 1
            return self._models[key]
        self.misses += 1
        model = loader_fn()
        self._models[key] = model
        return model

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "loaded_models": list(self._models.keys()),
        }


# ============================================================================
# 2. GRAPH CACHE — cache kết quả truy vấn Graph RAG (B1)
# ============================================================================

@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class GraphCache:
    """LRU + TTL cache cho kết quả hybrid_search()/truy vấn Cypher của Graph RAG.
    Key = normalize(query text) + top_k, để 2 câu hỏi giống hệt nhau (kể cả
    khác hoa/thường, khoảng trắng thừa) không phải truy vấn Neo4j lại.
    """

    def __init__(self, max_size: int = 500, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: "OrderedDict[str, _CacheEntry]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _normalize_key(query: str, top_k: int) -> str:
        norm = re.sub(r"\s+", " ", query.strip().lower())
        return hashlib.sha256(f"{norm}|{top_k}".encode("utf-8")).hexdigest()

    def get(self, query: str, top_k: int = 3) -> Optional[Any]:
        key = self._normalize_key(query, top_k)
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        if entry.expires_at < time.monotonic():
            del self._store[key]
            self.misses += 1
            return None
        self._store.move_to_end(key)  # LRU: vừa dùng -> đẩy lên gần đầu
        self.hits += 1
        return entry.value

    def set(self, query: str, top_k: int, value: Any) -> None:
        key = self._normalize_key(query, top_k)
        self._store[key] = _CacheEntry(value=value, expires_at=time.monotonic() + self.ttl_seconds)
        self._store.move_to_end(key)
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)  # evict item cũ nhất (LRU)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "current_size": len(self._store),
        }


async def cached_graph_search(kb, cache: GraphCache, query: str, top_k: int = 3) -> Any:
    """Wrapper: bọc quanh kb.hybrid_search() hiện có (agents.py) để thêm cache,
    KHÔNG cần sửa MockKnowledgeBase/Graph RAG thật — chỉ cần đổi lời gọi trong
    agents.py từ `self.kb.hybrid_search(text, top_k=3)` thành
    `await cached_graph_search(self.kb, self.graph_cache, text, top_k=3)`.

    QUAN TRỌNG: hàm này phải là `async def` và `await kb.hybrid_search(...)` vì
    KnowledgeBase.hybrid_search() thật (agents.py/knowledge_base.py, A2) là async.
    Nếu để dạng sync và không await, coroutine trả về sẽ bị lưu THẲNG vào cache
    (chưa chạy) -> lần cache-hit đầu tiên awaited nó thành công (may mắn 1 lần),
    nhưng cache-hit LẦN 2 trở đi sẽ crash "cannot reuse already awaited coroutine"
    vì 1 coroutine object chỉ await được đúng 1 lần. Phải await trước khi cache.set().
    """
    cached = cache.get(query, top_k)
    if cached is not None:
        return cached
    result = await kb.hybrid_search(query, top_k=top_k)
    cache.set(query, top_k, result)
    return result
class CachedKnowledgeBase:
    """Wrapper minh bạch quanh 1 KnowledgeBase bất kỳ (Neo4jKnowledgeBase/MockKnowledgeBase):
    thêm GraphCache (B2.3) mà KHÔNG cần sửa agents.py — interface hybrid_search() giữ nguyên,
    agents.py vẫn gọi kb.hybrid_search(...) như cũ, chỉ khác là kb truyền vào giờ là
    CachedKnowledgeBase thay vì Neo4jKnowledgeBase trực tiếp.
    """

    def __init__(self, kb: Any, cache: "GraphCache"):
        self._kb = kb
        self.cache = cache

    async def hybrid_search(self, query: str, top_k: int = 3) -> Any:
        return await cached_graph_search(self._kb, self.cache, query, top_k=top_k)

    def __getattr__(self, name: str) -> Any:
        # forward mọi attribute/method khác (nếu agents.py cần) sang kb gốc
        return getattr(self._kb, name)

# ============================================================================
# 3. SEMANTIC CACHE (bonus) — cache theo độ tương đồng ngữ nghĩa >= 0.95
# ============================================================================

def _default_bow_embed(text: str) -> dict[str, float]:
    """Embedding rất nhẹ (bag-of-words + IDF giả định đều nhau) dùng để DEMO
    không cần tải model ngoài. Khi tích hợp thật, truyền embed_fn = embedding
    model thật (vd. dùng chung với Graph RAG B1) vào SemanticCache(embed_fn=...).
    """
    tokens = re.findall(r"\w+", text.lower())
    vec: dict[str, float] = {}
    for t in tokens:
        vec[t] = vec.get(t, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def _cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = a.keys() & b.keys()
    return sum(a[k] * b[k] for k in keys)


def _cosine_dense(a: list[float], b: list[float]) -> float:
    """Cosine similarity cho dense vector (embedding model thật). embed_texts()
    trong B1/embeddings.py đã normalize_embeddings=True nên chỉ cần dot product."""
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


_B1_EMBEDDINGS_MODULE = None  # cache module đã nạp, tránh load lại model mỗi lần gọi


def _load_b1_embeddings_module():
    global _B1_EMBEDDINGS_MODULE
    if _B1_EMBEDDINGS_MODULE is not None:
        return _B1_EMBEDDINGS_MODULE

    import os
    import importlib.util

    _here = os.path.dirname(os.path.abspath(__file__))
    _b1_embeddings_path = os.path.abspath(os.path.join(_here, "..", "B1", "embeddings.py"))

    if not os.path.exists(_b1_embeddings_path):
        raise FileNotFoundError(
            f"Khong tim thay embeddings.py o: {_b1_embeddings_path}. "
            "Kiem tra lai cau truc thu muc B1/A2."
        )

    spec = importlib.util.spec_from_file_location("b1_embeddings_module", _b1_embeddings_path)
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    _B1_EMBEDDINGS_MODULE = _mod
    return _mod


def real_embed_fn(text: str) -> list[float]:
    """embed_fn THẬT dùng khi tích hợp production — tái sử dụng đúng embedding
    model (Qwen3-Embedding-0.6B, qua embed_one() trong B1/embeddings.py) đã dùng
    cho Graph RAG hybrid_search, đảm bảo Semantic Cache và Graph RAG cùng chung
    một không gian vector (nhất quán ngữ nghĩa). Module B1/embeddings.py được
    nạp bằng importlib theo ĐƯỜNG DẪN TUYỆT ĐỐI (không dùng sys.path/import
    tương đối) để tránh lỗi ModuleNotFoundError khi chạy trong venv/thư mục
    khác nhau, và chỉ nạp 1 lần (cache ở _B1_EMBEDDINGS_MODULE) để không load
    lại model mỗi lần gọi."""
    mod = _load_b1_embeddings_module()
    return mod.embed_one(text)


@dataclass
class _SemanticEntry:
    query: str
    embedding: Any
    response: Any
    expires_at: float


class SemanticCache:
    """Cache theo NGỮ NGHĨA: câu hỏi paraphrase khác chữ nhưng ý giống nhau
    (cosine similarity >= threshold) vẫn được coi là cache hit, không cần gọi
    lại Generator/Graph RAG. Đây là phần BONUS của B2.3.

    Ví dụ: "Wifi pass là gì vậy em?" và "Cho anh xin mật khẩu wifi với" —
    câu chữ khác hẳn nhưng ý giống nhau -> nên trả cùng 1 câu trả lời cache.
    """

    def __init__(
        self,
        threshold: float = 0.95,
        max_size: int = 200,
        ttl_seconds: float = 600.0,
        embed_fn: Callable[[str], Any] = real_embed_fn,
        similarity_fn: Callable[[Any, Any], float] = _cosine_dense,
    ):
        self.threshold = threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.embed_fn = embed_fn
        self.similarity_fn = similarity_fn
        self._entries: list[_SemanticEntry] = []
        self.hits = 0
        self.misses = 0

    def _purge_expired(self) -> None:
        now = time.monotonic()
        self._entries = [e for e in self._entries if e.expires_at >= now]

    def lookup(self, query: str) -> Optional[tuple[Any, float, str]]:
        """Trả về (response, similarity, matched_query) nếu có entry đủ giống,
        None nếu cache miss. Brute-force so sánh (chấp nhận được vì max_size nhỏ;
        nếu scale lớn hơn nên thay bằng FAISS/HNSW)."""
        self._purge_expired()
        if not self._entries:
            self.misses += 1
            return None

        query_emb = self.embed_fn(query)
        best_entry, best_sim = None, -1.0
        for entry in self._entries:
            sim = self.similarity_fn(query_emb, entry.embedding)
            if sim > best_sim:
                best_entry, best_sim = entry, sim

        if best_entry is not None and best_sim >= self.threshold:
            self.hits += 1
            return best_entry.response, best_sim, best_entry.query

        self.misses += 1
        return None

    def store(self, query: str, response: Any) -> None:
        self._purge_expired()
        embedding = self.embed_fn(query)
        self._entries.append(
            _SemanticEntry(
                query=query,
                embedding=embedding,
                response=response,
                expires_at=time.monotonic() + self.ttl_seconds,
            )
        )
        if len(self._entries) > self.max_size:
            self._entries.pop(0)  # evict entry cũ nhất

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "current_size": len(self._entries),
            "threshold": self.threshold,
        }


# ============================================================================
# DEMO
# ============================================================================

async def _demo():
    print("=" * 70)
    print("DEMO: Model Cache")
    print("=" * 70)
    model_cache = ModelCache()
    calls = {"n": 0}

    def fake_loader():
        calls["n"] += 1
        return f"<model handle #{calls['n']}>"

    m1 = model_cache.get_or_load("generator", fake_loader)
    m2 = model_cache.get_or_load("generator", fake_loader)
    print(f"  Lần 1: {m1} | Lần 2: {m2} | Loader chỉ được gọi thật {calls['n']} lần")
    print(f"  Stats: {model_cache.stats()}")
    assert calls["n"] == 1, "FAIL: model bị load lại!"
    print("  [PASS] Model chỉ load 1 lần dù gọi get_or_load 2 lần.\n")

    print("=" * 70)
    print("DEMO: Graph Cache (exact-match, TTL/LRU)")
    print("=" * 70)
    graph_cache = GraphCache(max_size=10, ttl_seconds=5)
    real_calls = {"n": 0}

    class FakeKB:
        async def hybrid_search(self, q, top_k=3):  # async để khớp đúng interface thật (agents.py/knowledge_base.py)
            real_calls["n"] += 1
            return [f"KQ cho '{q}' (top_k={top_k}, lần gọi thật #{real_calls['n']})"]

    kb = FakeKB()
    r1 = await cached_graph_search(kb, graph_cache, "Wifi pass là gì?", top_k=3)
    r2 = await cached_graph_search(kb, graph_cache, "Wifi pass là gì?", top_k=3)  # cache hit
    r3 = await cached_graph_search(kb, graph_cache, "  WIFI PASS LÀ GÌ?  ", top_k=3)  # normalize -> vẫn hit
    print(f"  r1={r1}\n  r2={r2}\n  r3={r3}")
    print(f"  Stats: {graph_cache.stats()}")
    assert real_calls["n"] == 1, "FAIL: Graph RAG bị gọi lại dù câu hỏi giống hệt!"
    print("  [PASS] Cache hit đúng cho câu hỏi trùng/khác hoa-thường.\n")

    print("=" * 70)
    print("DEMO: Semantic Cache (bonus, threshold=0.95 -- embedding THẬT Qwen3-Embedding)")
    print("=" * 70)
    # Dùng đúng embed_fn thật (real_embed_fn -> embed_one() của B1/embeddings.py,
    # cùng model Qwen3-Embedding-0.6B với Graph RAG) và threshold=0.95 đúng đề bài.
    # Model sẽ được tải lần đầu gọi (có thể mất vài giây / cần tải weight nếu
    # chưa cache local).
    sem_cache = SemanticCache(threshold=0.95, ttl_seconds=60)
    stored_query = "Wifi pass là gì vậy em?"
    sem_cache.store(stored_query, "Wifi pass là highlands2024 nhé.")

    q1 = "Cho anh xin pass wifi với"
    q2 = "Quán có bán bánh mì không?"

    # In ra similarity THẬT (raw cosine) để thấy rõ con số, không chỉ Hit/Miss
    stored_emb = sem_cache.embed_fn(stored_query)
    sim_q1 = sem_cache.similarity_fn(sem_cache.embed_fn(q1), stored_emb)
    sim_q2 = sem_cache.similarity_fn(sem_cache.embed_fn(q2), stored_emb)
    print(f"  Cosine('{q1}', '{stored_query}') = {sim_q1:.4f}")
    print(f"  Cosine('{q2}', '{stored_query}') = {sim_q2:.4f}")

    hit = sem_cache.lookup(q1)
    miss = sem_cache.lookup(q2)
    print(f"  Query gần giống (threshold=0.95, embedding thật) -> {hit}")
    print(f"  Query khác hẳn                                    -> {miss}")
    print(f"  Stats: {sem_cache.stats()}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_demo())