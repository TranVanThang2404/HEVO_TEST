"""
Embedding + Reranker wrapper cho Graph RAG (B1.2).

- Embedding model: đa ngôn ngữ (Việt + Anh), dùng cho menu_embedding và chunk_embedding.
- Reranker: cross-encoder, dùng ở bước Late Reranking của Hybrid Search Pipeline.

Thiết kế: bọc 2 model này thành 1 class dùng trực tiếp trong process (ingest.py,
hybrid_search.py import thẳng), ĐỒNG THỜI expose qua FastAPI (chạy `python embeddings.py`)
để thỏa yêu cầu đề bài "Serve qua TEI hoặc tự build API" — tự build API ở đây, không cần
dựng riêng TEI (Text Embeddings Inference) vốn nặng và khó chạy trên máy cấu hình yếu.
"""
from __future__ import annotations

import os
from functools import lru_cache

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer
    print(f"[embeddings] Đang nạp embedding model: {EMBED_MODEL_NAME} ...")
    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder
    print(f"[embeddings] Đang nạp reranker model: {RERANKER_MODEL_NAME} ...")
    return CrossEncoder(RERANKER_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Trả về vector embedding cho danh sách văn bản (dùng cho cả indexing lẫn query)."""
    model = _get_embedder()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_one(text: str) -> list[float]:
    return embed_texts([text])[0]


def rerank(query: str, candidates: list[str]) -> list[float]:
    """Late Reranking (B1.2): chấm điểm độ liên quan giữa query và từng candidate,
    trả về danh sách điểm số (cao hơn = liên quan hơn), cùng thứ tự với `candidates`."""
    if not candidates:
        return []
    model = _get_reranker()
    pairs = [(query, c) for c in candidates]
    scores = model.predict(pairs)
    return [float(s) for s in scores]


# ==================== Tự build API (thay thế TEI) ====================
if __name__ == "__main__":
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="Highlands RAG Embedding/Reranker Service")

    class EmbedRequest(BaseModel):
        texts: list[str]

    class RerankRequest(BaseModel):
        query: str
        candidates: list[str]

    @app.post("/embed")
    def api_embed(req: EmbedRequest):
        return {"embeddings": embed_texts(req.texts)}

    @app.post("/rerank")
    def api_rerank(req: RerankRequest):
        return {"scores": rerank(req.query, req.candidates)}

    @app.get("/health")
    def health():
        return {"status": "ok", "embed_model": EMBED_MODEL_NAME, "reranker_model": RERANKER_MODEL_NAME}

    print("Khởi động Embedding/Reranker API tại http://0.0.0.0:8100 (endpoints: /embed, /rerank, /health)")
    uvicorn.run(app, host="0.0.0.0", port=8100)