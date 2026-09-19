"""
Neo4jKnowledgeBase — nối KnowledgeBase interface (A2) vào Hybrid Search Pipeline
thật đã viết & benchmark PASS ở B1 (hybrid_search.py), thay cho MockKnowledgeBase.
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Optional

_B1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "B1"))
if _B1_DIR not in sys.path:
    sys.path.insert(0, _B1_DIR)

from neo4j import GraphDatabase  # noqa: E402
import hybrid_search as _hs  # noqa: E402  (B1/hybrid_search.py)

from knowledge_base import KnowledgeBase, MenuItem, RetrievedChunk  # noqa: E402


class Neo4jKnowledgeBase(KnowledgeBase):
    def __init__(self):
        self._driver = GraphDatabase.driver(_hs.NEO4J_URI, auth=(_hs.NEO4J_USER, _hs.NEO4J_PASSWORD))

    async def lookup_menu_item(self, name_query: str) -> Optional[MenuItem]:
        q = name_query.strip().lower()
        def _run():
            with self._driver.session() as session:
                return session.run(
                    """
                    MATCH (m:MenuItem)
                    WHERE toLower(m.name) CONTAINS $q OR $q CONTAINS toLower(m.name)
                    RETURN m.name AS name, m.price_vnd AS price_vnd,
                           m.category AS category, m.description AS description
                    LIMIT 1
                    """,
                    q=q,
                ).single()
        row = await asyncio.to_thread(_run)
        if row is None:
            return None
        return MenuItem(name=row["name"], price_vnd=row["price_vnd"],
                         category=row["category"], description=row["description"] or "")

    async def list_menu(self, category: Optional[str] = None) -> list[MenuItem]:
        def _run():
            with self._driver.session() as session:
                if category:
                    rows = session.run(
                        "MATCH (m:MenuItem {category: $cat}) RETURN m.name AS name, "
                        "m.price_vnd AS price_vnd, m.category AS category, m.description AS description",
                        cat=category,
                    )
                else:
                    rows = session.run(
                        "MATCH (m:MenuItem) RETURN m.name AS name, m.price_vnd AS price_vnd, "
                        "m.category AS category, m.description AS description"
                    )
                return list(rows)
        rows = await asyncio.to_thread(_run)
        return [MenuItem(name=r["name"], price_vnd=r["price_vnd"],
                          category=r["category"], description=r["description"] or "") for r in rows]

    async def hybrid_search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        results = await asyncio.to_thread(_hs.hybrid_search, query, top_k, True)
        out = []
        for r in results:
            source = "menu" if r.kind == "menu" else r.metadata.get("source", "chunk")
            out.append(RetrievedChunk(text=r.text, score=r.rerank_score, source=source))
        return out
