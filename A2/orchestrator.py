from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

from session_store import SessionStore
from request_queue import RequestQueue, RequestTimeoutError, with_retry
from knowledge_base import KnowledgeBase, MockKnowledgeBase
from neo4j_knowledge_base import Neo4jKnowledgeBase
from cache_layer import GraphCache, cached_graph_search
from agents import ConsultantAgent, EchoGeneratorClient, FAQAgent, GeneratorClient, OrderAgent, detect_language
from sglang_generator_client import SGLangGeneratorClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("orchestrator")


class RouterClient(ABC):
    @abstractmethod
    async def classify(self, text: str) -> str:
        """Trả về 1 trong 4 nhãn: order/consultant/faq/ignore (A1.1)."""
        ...


class MockRouterClient(RouterClient):
    _ORDER_KW = ["đặt", "order", "cho tôi", "cho anh", "cho em", "tính tiền", "bill", "thêm", "bớt"]
    _CONSULT_KW = ["gợi ý", "tư vấn", "recommend", "suggest", "món nào", "nên uống", "ngon"]
    _FAQ_KW = ["wifi", "giờ", "mở cửa", "đóng cửa", "địa chỉ", "gửi xe", "hours", "address", "open", "close"]

    async def classify(self, text: str) -> str:
        t = text.lower()
        if any(k in t for k in self._FAQ_KW):
            return "faq"
        if any(k in t for k in self._CONSULT_KW):
            return "consultant"
        if any(k in t for k in self._ORDER_KW):
            return "order"
        return "ignore"


class LocalSLMRouterClient(RouterClient):
    def __init__(self):
        import router_agent_fast as _r
        self._classify_intent = _r.classify_intent

    async def classify(self, text: str) -> str:
        loop = asyncio.get_running_loop()
        result, _latency_ms = await loop.run_in_executor(None, self._classify_intent, text)
        return result["action"]


_summary_generator: GeneratorClient = SGLangGeneratorClient()


async def summarize_history(old_summary: str, turns: list) -> str:
    turns_text = "\n".join(f"{t.role}: {t.text}" for t in turns)
    prompt = (
        "Tóm tắt ngắn gọn (2-3 câu) đoạn hội thoại sau, giữ lại thông tin quan trọng "
        f"(món đã gọi, sở thích khách...):\nTóm tắt cũ: {old_summary}\n{turns_text}"
    )
    return await _summary_generator.generate("Bạn là trợ lý tóm tắt hội thoại.", prompt)


class CachedKnowledgeBase(KnowledgeBase):
    def __init__(self, inner: KnowledgeBase, cache: GraphCache):
        self.inner = inner
        self.cache = cache

    async def lookup_menu_item(self, name_query: str):
        return await self.inner.lookup_menu_item(name_query)

    async def list_menu(self, category=None):
        return await self.inner.list_menu(category)

    async def hybrid_search(self, query: str, top_k: int = 3):
        return await cached_graph_search(self.inner, self.cache, query, top_k=top_k)




class MultiAgentOrchestrator:
    def __init__(self, router: RouterClient, session_store: SessionStore, queue: RequestQueue):
        self.router = router
        self.sessions = session_store
        self.queue = queue
        neo4j_kb = Neo4jKnowledgeBase()
        self.graph_cache = GraphCache()                       # B2.3
        kb = CachedKnowledgeBase(neo4j_kb, self.graph_cache)   # bọc cache, agents.py không cần sửa
        generator: GeneratorClient = SGLangGeneratorClient()
        self.agents = {
            "order": OrderAgent(kb, generator),
            "consultant": ConsultantAgent(kb, generator),
            "faq": FAQAgent(kb, generator),
        }

    async def handle_message(self, session_id: str, user_text: str) -> dict:
        @with_retry(max_retries=3)
        async def _classify():
            return await self.router.classify(user_text)

        try:
            intent = await self.queue.run(_classify)
        except RequestTimeoutError as e:
            return {
                "agent": "router",
                "intent": "timeout",
                "session_id": session_id,
                "language": detect_language(user_text),
                "reply": str(e),
                "meta": {"error": "timeout"},
            }

        summary, history = self.sessions.get_context(session_id)
        await self.sessions.add_turn(session_id, "user", user_text)

        if intent == "ignore" or intent not in self.agents:
            lang = detect_language(user_text)
            reply = (
                "Xin lỗi, mình chưa rõ yêu cầu của bạn, bạn có thể nói rõ hơn được không?"
                if lang == "vi"
                else "Sorry, I couldn't quite understand your request — could you rephrase?"
            )
            result = {"agent": "ignore", "language": lang, "reply": reply, "meta": {}}
        else:
            agent = self.agents[intent]

            @with_retry(max_retries=3)
            async def _handle():
                return await agent.handle(user_text, summary, history)

            try:
                result = await self.queue.run(_handle)
            except RequestTimeoutError as e:
                result = {
                    "agent": intent,
                    "language": detect_language(user_text),
                    "reply": str(e),
                    "meta": {"error": "timeout"},
                }

        await self.sessions.add_turn(session_id, "assistant", result["reply"])
        result["intent"] = intent
        result["session_id"] = session_id
        return result