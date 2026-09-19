from __future__ import annotations
 
import re
from abc import ABC, abstractmethod
from typing import Optional
 
from knowledge_base import KnowledgeBase, MenuItem, RetrievedChunk
 
_VI_CHARS = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)
_VI_COMMON_WORDS = {"anh", "chi", "em", "toi", "minh", "gia", "quan", "mon", "gio", "cho", "khong"}
 
 
def detect_language(text: str) -> str:
    """Heuristic đơn giản: có ký tự có dấu tiếng Việt hoặc từ khoá tiếng Việt phổ biến -> vi, ngược lại en."""
    if _VI_CHARS.search(text):
        return "vi"
    if any(w in text.lower().split() for w in _VI_COMMON_WORDS):
        return "vi"
    return "en"
 
 
class GeneratorClient(ABC):
    """Interface gọi LLM sinh câu trả lời tự nhiên.
    B2 sẽ implement bản gọi Generator thật qua SGLang/vLLM (HTTP + streaming SSE)."""
 
    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> str: ...
 
 
class EchoGeneratorClient(GeneratorClient):
    """Bản triển khai placeholder KHÔNG cần GPU/model — ghép template có sẵn.
    Dùng để test luồng multi-agent (A2) độc lập với hạ tầng serving (B2) chưa xây."""
 
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        role_line = system_prompt.splitlines()[0]
        return f"[{role_line}] {user_prompt}"
 
 
class BaseAgent(ABC):
    name: str
    system_prompt_vi: str
    system_prompt_en: str
 
    def __init__(self, kb: KnowledgeBase, generator: GeneratorClient):
        self.kb = kb
        self.generator = generator
 
    def system_prompt_for(self, lang: str) -> str:
        return self.system_prompt_vi if lang == "vi" else self.system_prompt_en
 
    @abstractmethod
    async def handle(self, user_text: str, summary: str, history: list) -> dict:
        """Trả về dict chuẩn: {"agent": self.name, "language": lang, "reply": str, "meta": {...}}"""
        ...
 
 
class OrderAgent(BaseAgent):
    name = "order"
    system_prompt_vi = (
        "Bạn là Order Agent của Highlands Coffee. Nhiệm vụ: tiếp nhận đặt hàng, "
        "thêm/bớt món, tính tiền chính xác theo đúng giá trong Menu. Luôn xác nhận "
        "lại món + giá trước khi chốt đơn. "
        "QUAN TRỌNG: nếu món khách yêu cầu KHÔNG khớp với bất kỳ món nào trong Menu "
        "được cung cấp (kể cả sai size, sai tên), PHẢI từ chối lịch sự, nói rõ quán "
        "không có món/size đó. Nếu muốn gợi ý món thay thế, CHỈ được chọn NGUYÊN VĂN "
        "tên món và giá đã xuất hiện trong danh sách Menu được cung cấp ở trên — TUYỆT "
        "ĐỐI KHÔNG được tự tạo ra tên món mới hoặc giá mới không có trong Menu. Nếu "
        "không chắc món nào trong Menu là hợp lý để gợi ý, chỉ cần từ chối và hỏi khách "
        "muốn xem lại menu không, không cần đoán. Trả lời ngắn gọn, thân thiện, đúng tiếng Việt."
    )
    system_prompt_en = (
        "You are the Order Agent for Highlands Coffee. Handle orders, add/remove items, "
        "and compute the bill exactly from the Menu prices. Always confirm items + price "
        "before finalizing. "
        "IMPORTANT: if the requested item does NOT match any item in the provided Menu "
        "(including wrong size or wrong name), you MUST politely decline and state clearly "
        "that the item/size is not available. If you suggest an alternative, you MUST pick "
        "the EXACT item name and price as they literally appear in the provided Menu list — "
        "NEVER invent a new item name or price not present in the Menu. If unsure which real "
        "item to suggest, simply decline and ask if they'd like to see the menu, rather than "
        "guessing. Reply briefly and friendly, in English."
    )
 
    async def handle(self, user_text: str, summary: str, history: list) -> dict:
        lang = detect_language(user_text)
        item: Optional[MenuItem] = await self.kb.lookup_menu_item(user_text)
        if item:
            context = f"Món: {item.name} - Giá: {item.price_vnd:,}đ - Mô tả: {item.description}"
        else:
            menu = await self.kb.list_menu()
            context = "Chưa xác định món cụ thể trong câu khách. Menu hiện có: " + ", ".join(
                f"{m.name} ({m.price_vnd:,}đ)" for m in menu
            )
        prompt = f"Ngữ cảnh Menu: {context}\nLịch sử tóm tắt: {summary}\nKhách nói: {user_text}"
        reply = await self.generator.generate(self.system_prompt_for(lang), prompt)
        return {
            "agent": self.name,
            "language": lang,
            "reply": reply,
            "meta": {"matched_item": item.name if item else None},
        }
 
 
class ConsultantAgent(BaseAgent):
    name = "consultant"
    system_prompt_vi = (
        "Bạn là Consultant Agent của Highlands Coffee. Nhiệm vụ: tư vấn, gợi ý món dựa "
        "trên khẩu vị, ngân sách, thời tiết khách đề cập, dùng RAG lấy từ Menu + FAQ. "
        "Chỉ gợi ý món CÓ THẬT trong context được cung cấp, không bịa. Trả lời tiếng Việt."
    )
    system_prompt_en = (
        "You are the Consultant Agent for Highlands Coffee. Recommend items based on taste, "
        "budget, or weather using the retrieved Menu + FAQ context (RAG). Only recommend "
        "items that actually appear in the provided context. Reply in English."
    )
 
    async def handle(self, user_text: str, summary: str, history: list) -> dict:
        lang = detect_language(user_text)
        chunks: list[RetrievedChunk] = await self.kb.hybrid_search(user_text, top_k=3)
        context = "\n".join(f"- {c.text}" for c in chunks)
        prompt = f"Context (RAG - Menu+FAQ):\n{context}\nLịch sử tóm tắt: {summary}\nKhách nói: {user_text}"
        reply = await self.generator.generate(self.system_prompt_for(lang), prompt)
        return {
            "agent": self.name,
            "language": lang,
            "reply": reply,
            "meta": {"retrieved": [c.text for c in chunks]},
        }
 
 
class FAQAgent(BaseAgent):
    name = "faq"
    system_prompt_vi = (
        "Bạn là FAQ Agent của Highlands Coffee. Nhiệm vụ: trả lời các câu hỏi chung về "
        "quán (wifi, giờ mở cửa, chỗ gửi xe, chính sách...) CHỈ dựa trên context truy xuất "
        "được (Hybrid RAG: Dual-Domain Search -> Graph Expansion -> Late Reranking). "
        "Không bịa thông tin ngoài context. Trả lời tiếng Việt."
    )
    system_prompt_en = (
        "You are the FAQ Agent for Highlands Coffee. Answer general questions about the shop "
        "(wifi, hours, parking, policies...) ONLY using the retrieved context (Hybrid RAG: "
        "Dual-Domain Search -> Graph Expansion -> Late Reranking). Never invent information "
        "outside the given context. Reply in English."
    )
 
    async def handle(self, user_text: str, summary: str, history: list) -> dict:
        lang = detect_language(user_text)
        chunks: list[RetrievedChunk] = await self.kb.hybrid_search(user_text, top_k=3)
        context = "\n".join(f"- {c.text}" for c in chunks)
        prompt = f"Context (Hybrid RAG):\n{context}\nLịch sử tóm tắt: {summary}\nKhách hỏi: {user_text}"
        reply = await self.generator.generate(self.system_prompt_for(lang), prompt)
        return {
            "agent": self.name,
            "language": lang,
            "reply": reply,
            "meta": {"retrieved": [c.text for c in chunks]},
        }
 