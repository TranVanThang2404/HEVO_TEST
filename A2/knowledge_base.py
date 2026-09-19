
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class MenuItem:
    name: str
    price_vnd: int
    category: str
    description: str = ""


@dataclass
class RetrievedChunk:
    text: str
    score: float
    source: str = "faq"  # "faq" | "document" | "menu" | "none"


class KnowledgeBase(ABC):
    """Interface chung — sẽ được thay bằng Neo4jKnowledgeBase ở Module B1."""

    @abstractmethod
    async def lookup_menu_item(self, name_query: str) -> Optional[MenuItem]: ...

    @abstractmethod
    async def list_menu(self, category: Optional[str] = None) -> list[MenuItem]: ...

    @abstractmethod
    async def hybrid_search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Dual-Domain Search -> Graph Expansion -> Late Reranking (B1.2).
        Bản mock ở đây chỉ làm keyword-match đơn giản để Agent có context trả lời."""
        ...


class MockKnowledgeBase(KnowledgeBase):
    def __init__(self):
        self._menu = [
            MenuItem("Bạc xỉu", 45000, "coffee", "Cà phê sữa đá đặc trưng, ít đắng, béo nhẹ."),
            MenuItem("Cà phê sữa đá", 39000, "coffee", "Đậm vị cà phê phin truyền thống."),
            MenuItem("Trà đào cam sả", 49000, "tea", "Trà trái cây thanh mát, ít ngọt có thể yêu cầu."),
            MenuItem("Freeze trà xanh", 55000, "freeze", "Đá xay trà xanh, vị ngọt béo, mát lạnh."),
            MenuItem("Latte đá", 49000, "coffee", "Cà phê espresso pha sữa tươi, vị nhẹ béo."),
        ]
        self._faq = [
            "Quán mở cửa từ 6:30 sáng đến 22:00 mỗi ngày, kể cả cuối tuần.",
            "Wifi miễn phí cho khách, tên wifi 'Highlands_FreeWifi', mật khẩu 'highlands123'.",
            "Quán có chỗ giữ xe máy miễn phí phía sau, có bảo vệ trông xe.",
            "Có thể đặt bàn trước qua hotline hoặc app, đặt bàn tiệc sinh nhật cần báo trước ít nhất 2 tiếng.",
            "Highlands Coffee chấp nhận thanh toán tiền mặt, thẻ, và ví điện tử (Momo, ZaloPay, VNPay).",
        ]

    async def lookup_menu_item(self, name_query: str) -> Optional[MenuItem]:
        q = name_query.lower()
        for item in self._menu:
            if q in item.name.lower() or item.name.lower() in q:
                return item
        return None

    async def list_menu(self, category: Optional[str] = None) -> list[MenuItem]:
        if category is None:
            return list(self._menu)
        return [m for m in self._menu if m.category == category]

    async def hybrid_search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        q = query.lower()
        candidates: list[RetrievedChunk] = []
        for fact in self._faq:
            overlap = len(set(q.split()) & set(fact.lower().split()))
            if overlap > 0:
                candidates.append(RetrievedChunk(text=fact, score=overlap / 10, source="faq"))
        for item in self._menu:
            overlap = len(set(q.split()) & set(item.description.lower().split()))
            if overlap > 0 or item.name.lower() in q:
                candidates.append(
                    RetrievedChunk(
                        text=f"{item.name} ({item.price_vnd:,}đ) - {item.description}",
                        score=(overlap + (2 if item.name.lower() in q else 0)) / 10,
                        source="menu",
                    )
                )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k] if candidates else [
            RetrievedChunk(text="(Không tìm thấy thông tin liên quan trong dữ liệu quán.)", score=0.0, source="none")
        ]