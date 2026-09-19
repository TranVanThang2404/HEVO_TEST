"""
SGLangGeneratorClient — bản triển khai thật của GeneratorClient (B2.1), gọi
Generator SGLang server qua HTTP (/v1/chat/completions), thay cho EchoGeneratorClient.
"""
from __future__ import annotations

import os

import httpx

from agents import GeneratorClient
from request_queue import with_retry

GENERATOR_URL = os.getenv("GENERATOR_URL", "http://127.0.0.1:30001/v1/chat/completions")
GENERATOR_TIMEOUT = float(os.getenv("GENERATOR_TIMEOUT", "30.0"))


class SGLangGeneratorClient(GeneratorClient):
    def __init__(self, url: str = GENERATOR_URL, timeout: float = GENERATOR_TIMEOUT):
        self._url = url
        self._client = httpx.AsyncClient(timeout=timeout)

    @with_retry(max_retries=3, exceptions=(httpx.HTTPError,))
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": "generator",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 256,
        }
        resp = await self._client.post(self._url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
