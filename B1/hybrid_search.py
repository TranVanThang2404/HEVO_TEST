"""
B1.2 — Hybrid Search Pipeline (đúng 3 bước theo đề bài):

    Dual-Domain Search  ->  Graph Expansion  ->  Late Reranking

1. Dual-Domain Search:
   Tìm song song trên 2 vector index Neo4j (menu_embedding trên MenuItem,
   chunk_embedding trên Chunk), lấy top-k ứng viên thô từ mỗi domain bằng
   cosine similarity (native Neo4j Vector Index, `db.index.vector.queryNodes`).

2. Graph Expansion:
   Với mỗi Chunk ứng viên, mở rộng ngữ cảnh bằng cách duyệt quan hệ
   NEXT/PREV (chunk liền kề trong tài liệu gốc) và MENTIONS (Entity chunk
   nhắc tới) -> tìm thêm các Chunk khác cũng MENTIONS cùng Entity đó.
   Mục đích: kéo thêm ngữ cảnh liên quan mà pure vector search một mình bỏ
   sót (khác section/khác câu nhưng cùng nói về 1 thực thể).

3. Late Reranking:
   Dùng cross-encoder (embeddings.py::rerank) chấm lại toàn bộ tập ứng viên
   (thô + mở rộng) theo đúng câu query gốc, chỉ giữ lại những kết quả có
   relevance score >= RERANK_THRESHOLD (0.7 theo đề bài), sắp xếp giảm dần.

Dùng trực tiếp trong process (orchestrator/KnowledgeBase sẽ import hàm
`hybrid_search(query, top_k)` này), và cũng có thể chạy độc lập để test nhanh
qua CLI: `python hybrid_search.py "wifi pass là gì"`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from neo4j import GraphDatabase

from embeddings import embed_one, rerank

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "highlands123")

RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.7"))
DUAL_DOMAIN_TOPK = int(os.getenv("DUAL_DOMAIN_TOPK", "8"))   # top-k thô mỗi domain (menu / chunk)
EXPANSION_TOPK_ENTITY = int(os.getenv("EXPANSION_TOPK_ENTITY", "3"))  # số chunk mở rộng thêm / entity

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


@dataclass
class SearchResult:
    kind: str            # "menu" | "chunk"
    node_id: str         # MenuItem.name hoặc Chunk.chunk_id
    text: str            # nội dung để hiển thị / rerank
    vector_score: float = 0.0
    rerank_score: float = 0.0
    via_expansion: bool = False
    metadata: dict = field(default_factory=dict)


# ==================== Bước 1: Dual-Domain Search ====================

def _search_menu(query_vec: list[float], top_k: int) -> list[SearchResult]:
    with driver.session() as session:
        rows = session.run(
            """
            CALL db.index.vector.queryNodes('menu_embedding', $top_k, $vec)
            YIELD node, score
            RETURN node.name AS name, node.description AS description,
                   node.price_vnd AS price_vnd, node.category AS category, score
            """,
            top_k=top_k, vec=query_vec,
        )
        return [
            SearchResult(
                kind="menu", node_id=r["name"],
                text=f"{r['name']}: {r['description']} ({r['price_vnd']}đ, {r['category']})",
                vector_score=r["score"],
                metadata={"price_vnd": r["price_vnd"], "category": r["category"]},
            )
            for r in rows
        ]


def _search_chunks(query_vec: list[float], top_k: int) -> list[SearchResult]:
    with driver.session() as session:
        rows = session.run(
            """
            CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $vec)
            YIELD node, score
            RETURN node.chunk_id AS chunk_id, node.text AS text, node.source AS source,
                   node.section AS section, node.category AS category, score
            """,
            top_k=top_k, vec=query_vec,
        )
        return [
            SearchResult(
                kind="chunk", node_id=r["chunk_id"], text=r["text"],
                vector_score=r["score"],
                metadata={"source": r["source"], "section": r["section"], "category": r["category"]},
            )
            for r in rows
        ]


def dual_domain_search(query: str, top_k: int = DUAL_DOMAIN_TOPK) -> list[SearchResult]:
    query_vec = embed_one(query)
    return _search_menu(query_vec, top_k) + _search_chunks(query_vec, top_k)


# ==================== Bước 2: Graph Expansion ====================

def graph_expansion(candidates: list[SearchResult]) -> list[SearchResult]:
    """Mở rộng ngữ cảnh qua NEXT/PREV (chunk liền kề) và MENTIONS (entity chung)."""
    chunk_ids = [c.node_id for c in candidates if c.kind == "chunk"]
    if not chunk_ids:
        return []

    expanded: list[SearchResult] = []
    seen_ids = {c.node_id for c in candidates}

    with driver.session() as session:
        # (a) Chunk liền kề: NEXT phía sau và phía trước (PREV = đảo chiều NEXT)
        rows = session.run(
            """
            UNWIND $ids AS cid
            MATCH (c:Chunk {chunk_id: cid})
            OPTIONAL MATCH (c)-[:NEXT]->(nxt:Chunk)
            OPTIONAL MATCH (prv:Chunk)-[:NEXT]->(c)
            RETURN cid,
                   collect(DISTINCT nxt {.chunk_id, .text, .source, .section, .category}) AS next_chunks,
                   collect(DISTINCT prv {.chunk_id, .text, .source, .section, .category}) AS prev_chunks
            """,
            ids=chunk_ids,
        )
        for r in rows:
            for neigh in (r["next_chunks"] or []) + (r["prev_chunks"] or []):
                if not neigh or neigh.get("chunk_id") is None or neigh["chunk_id"] in seen_ids:
                    continue
                seen_ids.add(neigh["chunk_id"])
                expanded.append(SearchResult(
                    kind="chunk", node_id=neigh["chunk_id"], text=neigh["text"],
                    via_expansion=True,
                    metadata={"source": neigh.get("source"), "section": neigh.get("section"),
                              "category": neigh.get("category"), "expansion_type": "adjacent"},
                ))

        # (b) Entity chung: các Chunk khác cũng MENTIONS cùng Entity với chunk ứng viên
        rows = session.run(
            """
            UNWIND $ids AS cid
            MATCH (c:Chunk {chunk_id: cid})-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(other:Chunk)
            WHERE other.chunk_id <> cid
            WITH other, count(DISTINCT e) AS shared_entities
            ORDER BY shared_entities DESC
            RETURN DISTINCT other {.chunk_id, .text, .source, .section, .category} AS chunk, shared_entities
            LIMIT $limit
            """,
            ids=chunk_ids, limit=EXPANSION_TOPK_ENTITY * len(chunk_ids),
        )
        for r in rows:
            chunk = r["chunk"]
            if chunk["chunk_id"] in seen_ids:
                continue
            seen_ids.add(chunk["chunk_id"])
            expanded.append(SearchResult(
                kind="chunk", node_id=chunk["chunk_id"], text=chunk["text"],
                via_expansion=True,
                metadata={"source": chunk.get("source"), "section": chunk.get("section"),
                          "category": chunk.get("category"), "expansion_type": "shared_entity",
                          "shared_entities": r["shared_entities"]},
            ))

    return expanded


# ==================== Bước 3: Late Reranking ====================

def late_reranking(query: str, candidates: list[SearchResult], threshold: float = RERANK_THRESHOLD,
                    top_k: int = 5) -> list[SearchResult]:
    if not candidates:
        return []
    texts = [c.text for c in candidates]
    scores = rerank(query, texts)
    for c, s in zip(candidates, scores):
        c.rerank_score = s
    passed = [c for c in candidates if c.rerank_score >= threshold]
    passed.sort(key=lambda c: c.rerank_score, reverse=True)
    return passed[:top_k]


# ==================== Pipeline đầy đủ ====================

def hybrid_search(query: str, top_k: int = 5, use_expansion: bool = True) -> list[SearchResult]:
    stage1 = dual_domain_search(query)
    candidates = list(stage1)
    if use_expansion:
        candidates += graph_expansion(stage1)
    return late_reranking(query, candidates, top_k=top_k)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "wifi pass là gì vậy"
    print(f"Query: {q!r}\n")
    results = hybrid_search(q)
    if not results:
        print("Không có kết quả nào vượt ngưỡng rerank >= "
              f"{RERANK_THRESHOLD} — thử hạ RERANK_THRESHOLD hoặc kiểm tra dữ liệu đã ingest chưa.")
    for i, r in enumerate(results, 1):
        tag = "[mở rộng]" if r.via_expansion else "[gốc]"
        print(f"{i}. {tag} ({r.kind}, rerank={r.rerank_score:.3f}) {r.text[:120]}")