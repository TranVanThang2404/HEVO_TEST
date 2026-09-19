"""
SGLangRouterClient — bản triển khai thật của RouterClient (A1.1), gọi Router
SGLang server (port 30000, model router_awq_v3) qua API native /generate với
return_logprob=True, TÁI SỬ DỤNG đúng phương pháp logit-scoring đã verify
(test_router_logit_scoring.py, 4/4 đúng) — KHÔNG tự load model vào GPU (chỉ
nạp tokenizer, chạy CPU, không tốn VRAM) nên an toàn khi chạy song song với
2 server SGLang đã chiếm gần hết 6GB VRAM.
"""
from __future__ import annotations

import os

import httpx
from transformers import AutoTokenizer

from orchestrator import RouterClient

ROUTER_URL = os.getenv("ROUTER_URL", "http://127.0.0.1:30000/generate")
ROUTER_TOKENIZER_DIR = os.getenv("ROUTER_TOKENIZER_DIR", "/mnt/d/HEVO/A1/router_awq_v3")
LABELS = ["order", "consultant", "faq", "ignore"]

SYSTEM_PROMPT = """Bạn là Router Agent của Highlands Coffee. Khách có thể nói tiếng Việt hoặc tiếng Anh.
Phân loại ý định của khách hàng thành 1 trong 4 loại, bất kể ngôn ngữ khách dùng:
- "order": Đặt hàng, tính tiền, gọi thêm món. Ví dụ: "Cho 1 ly bạc xỉu size M", "Tính tiền giúp em", "I'd like a large iced latte".
- "consultant": Nhờ tư vấn, gợi ý món. Ví dụ: "Có món nào ngọt mà không quá béo không?", "What do you recommend for a hot day?".
- "faq": Hỏi thông tin CỦA QUÁN (wifi, giờ mở cửa, địa chỉ, chỗ gửi xe...). Chỉ chọn faq khi câu hỏi thật sự hỏi về quán. Ví dụ: "Wifi pass gì vậy em?", "What time do you close?".
- "ignore": Câu nói KHÔNG mang yêu cầu/câu hỏi nào tới quán — chào hỏi vu vơ, test mic/kết nối, tiếng ồn, cười đùa, gọi tên suông. Đây KHÔNG phải faq dù có dạng câu hỏi. Ví dụ: "Alo alo test mic", "haha".
Trả lời đúng 1 nhãn duy nhất trong 4 nhãn trên."""


def _common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class SGLangRouterClient(RouterClient):
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(ROUTER_TOKENIZER_DIR)
        nl_ids = self._tokenizer.encode("\n", add_special_tokens=False)
        self._label_ids = []
        for lbl in LABELS:
            combo = self._tokenizer.encode("\n" + lbl, add_special_tokens=False)
            common = _common_prefix_len(nl_ids, combo)
            self._label_ids.append(combo[common:])
        self._client = httpx.AsyncClient(timeout=10.0)

    async def classify(self, text: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
        prompt_text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = self._tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        scores = []
        for lbl_ids in self._label_ids:
            full_ids = prompt_ids + lbl_ids
            resp = await self._client.post(ROUTER_URL, json={
                "input_ids": full_ids,
                "sampling_params": {"max_new_tokens": 0},
                "return_logprob": True,
                "logprob_start_len": 0,
            })
            resp.raise_for_status()
            data = resp.json()
            logprobs = data["meta_info"]["input_token_logprobs"]
            tail = logprobs[-len(lbl_ids):]
            total = sum(lp[0] for lp in tail if lp[0] is not None)
            scores.append(total / len(lbl_ids))

        return LABELS[scores.index(max(scores))]
