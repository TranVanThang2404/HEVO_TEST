"""
B2.2 — SSE Streaming API cho Generator (LLM Serving & Optimization).

Yêu cầu đề bài (B2.2):
  1. True streaming qua Server-Sent Events (SSE) — không đợi sinh xong cả câu.
  2. Xử lý đúng sentinel "[DONE]" theo đúng convention của OpenAI streaming API
     (literal string "data: [DONE]\n\n", KHÔNG bọc JSON — client phải check
     chuỗi này TRƯỚC khi json.loads, nếu không sẽ crash).
  3. Clause-level streaming riêng cho pipeline TTS — gom token thành từng
     mệnh đề/câu hoàn chỉnh (kết thúc bằng . ! ? ; : hoặc xuống dòng) thay vì
     đọc từng từ rời rạc (giật, không tự nhiên) hoặc đợi hết cả đoạn dài mới đọc
     (trễ, TTFT cho TTS cao).

Thiết kế: 1 endpoint SSE phát ra 2 loại event xen kẽ:
  - event: token   -> từng mảnh text thô, dùng để hiển thị hiệu ứng "gõ chữ" trên UI.
  - event: clause  -> từng câu/mệnh đề hoàn chỉnh, dùng để đẩy vào TTS đọc.
  - Kết thúc luôn bằng "data: [DONE]\n\n" (không có event: name).

Generator client là pluggable: đang để mặc định "echo" (giả lập, không cần
model thật) để có thể test luồng SSE/clause-splitting NGAY, độc lập với việc
quantize AWQ đang fix. Khi model AWQ Generator chạy đúng qua SGLang (port 30001),
chỉ cần đổi GENERATOR_MODE = "sglang" là dùng model thật, không cần sửa gì khác.

Cài đặt (venv nào cũng được, ví dụ llama-env hoặc serve-env):
    pip install fastapi uvicorn httpx --break-system-packages

Chạy:
    uvicorn api_server:app --host 0.0.0.0 --port 8080 --reload

Test:
    python test_sse_client.py
"""
import asyncio
import json
import time
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="B2.2 SSE Streaming API")

# ---------------------------------------------------------------------------
# Generator client (pluggable): "echo" để test ngay, "sglang" khi model đã fix.
# ---------------------------------------------------------------------------

GENERATOR_MODE = "sglang"  # đổi thành "sglang" khi Generator AWQ đã quantize lại đúng
SGLANG_GENERATOR_URL = "http://127.0.0.1:30001/v1/chat/completions"


async def stream_from_echo(prompt: str) -> AsyncIterator[str]:
    """Generator giả lập — để test SSE/clause-splitting mà không cần chờ model thật."""
    fake_reply = (
        "Dạ chào anh chị! Quán mình có bán cà phê sữa đá, size L giá 35 nghìn. "
        "Anh chị muốn thêm món nào không ạ? Wifi pass là highlands2024 nhé. "
        "Cảm ơn anh chị đã ghé quán!"
    )
    for word in fake_reply.split(" "):
        await asyncio.sleep(0.03)  # giả lập độ trễ sinh token của LLM thật
        yield word + " "


async def stream_from_sglang(prompt: str) -> AsyncIterator[str]:
    """Generator thật — gọi thẳng SGLang Generator server (OpenAI-compatible, stream=True)."""
    payload = {
        "model": "generator",
        "messages": [
            {
                "role": "system",
                "content": "Bạn là nhân viên tư vấn quán cà phê Highlands, trả lời ngắn gọn, thân thiện, bằng tiếng Việt.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "max_tokens": 256,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", SGLANG_GENERATOR_URL, json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    yield delta


def get_generator_stream(prompt: str) -> AsyncIterator[str]:
    if GENERATOR_MODE == "sglang":
        return stream_from_sglang(prompt)
    return stream_from_echo(prompt)


# ---------------------------------------------------------------------------
# Clause splitting cho TTS
# ---------------------------------------------------------------------------

CLAUSE_ENDERS = set(".!?;\n")


class ClauseBuffer:
    """Gom token thành từng mệnh đề/câu hoàn chỉnh để đẩy qua TTS mượt hơn.

    Không đọc từng từ rời rạc (giật cục), cũng không đợi hết cả đoạn dài mới
    đọc (trễ). Cắt ngay khi gặp dấu kết thúc mệnh đề.
    """

    def __init__(self):
        self._buf = ""

    def feed(self, token: str) -> Optional[str]:
        self._buf += token
        cut_idx = None
        for i, ch in enumerate(self._buf):
            if ch in CLAUSE_ENDERS:
                cut_idx = i
                break
        if cut_idx is None:
            return None
        clause = self._buf[: cut_idx + 1].strip()
        self._buf = self._buf[cut_idx + 1:]
        return clause if clause else None

    def flush(self) -> Optional[str]:
        remaining = self._buf.strip()
        self._buf = ""
        return remaining if remaining else None


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


def sse_event(data: dict, event: Optional[str] = None) -> str:
    lines = []
    if event:
        lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data, ensure_ascii=False)}")
    lines.append("")
    return "\n".join(lines) + "\n"


async def sse_generator(req: ChatRequest) -> AsyncIterator[str]:
    clause_buf = ClauseBuffer()
    request_id = f"chatcmpl-{int(time.time() * 1000)}"

    try:
        async for token in get_generator_stream(req.message):
            # (1) Luồng token thô — UI hiển thị hiệu ứng gõ chữ.
            yield sse_event(
                {"id": request_id, "object": "chat.completion.chunk",
                 "delta": {"content": token}},
                event="token",
            )

            # (2) Luồng clause — đẩy vào TTS, đọc theo từng câu/mệnh đề trọn vẹn.
            clause = clause_buf.feed(token)
            if clause:
                yield sse_event({"id": request_id, "clause": clause}, event="clause")

        remaining = clause_buf.flush()
        if remaining:
            yield sse_event({"id": request_id, "clause": remaining}, event="clause")

    except Exception as e:
        # Lỗi giữa chừng vẫn phải đóng stream đúng cách, không để client treo vô hạn.
        yield sse_event({"error": str(e)}, event="error")

    # Sentinel kết thúc — ĐÚNG CHUẨN OpenAI: literal string, KHÔNG bọc JSON.
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        sse_generator(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # tắt buffering nếu sau này có nginx đứng trước
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "generator_mode": GENERATOR_MODE}