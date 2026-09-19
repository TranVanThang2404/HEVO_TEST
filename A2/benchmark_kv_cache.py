"""
B2.3 — Verify tầng KV Cache.

KV Cache KHÔNG tự viết ở application layer — nó là tính năng built-in của
SGLang (radix-tree prefix cache), được bật bằng flag --schedule-policy lpm
(longest-prefix-match) lúc launch server. Script này chỉ dùng để CHỨNG MINH
nó đang hoạt động: gửi 2 request có chung 1 prefix DÀI (system prompt +
few-shot cố định), phần suffix (câu hỏi cuối) khác nhau. Nếu KV cache hoạt
động đúng, request thứ 2 sẽ có TTFT thấp hơn rõ rệt vì phần prefix trùng
không phải tính lại từ đầu (chỉ cần tính KV cho phần suffix mới).

Yêu cầu: Generator server (SGLang) đang chạy ở port 30001.

Chạy:
    python3 benchmark_kv_cache.py
"""
import time

import httpx

GENERATOR_URL = "http://127.0.0.1:30001/v1/chat/completions"

# Prefix CỐ ĐỊNH, dài, giống hệt nhau giữa 2 request -> để tận dụng prefix cache.
LONG_SYSTEM_PROMPT = (
    "Bạn là nhân viên tư vấn của quán cà phê Highlands Coffee. "
    "Menu quán gồm: Cà phê sữa đá (29k/35k/39k tuỳ size S/M/L), "
    "Bạc xỉu (32k/38k/42k), Trà đào cam sả (45k/49k/55k), "
    "Trà sữa trân châu (39k/45k/49k), Bánh mì que (15k/cái), "
    "Freeze trà xanh (55k/59k/65k). Quán mở cửa 6h30 - 22h hằng ngày, "
    "có wifi miễn phí, chỗ đậu xe máy phía sau quán, giao hàng trong bán kính 3km "
    "phí ship 15k. Luôn trả lời ngắn gọn, thân thiện, đúng thông tin trên, "
    "không bịa thêm món hoặc giá không có trong danh sách."
)

QUERIES = [
    "Cho anh 1 ly cà phê sữa đá size L.",
    "Quán có bán bánh mì que không, giá bao nhiêu?",
]


def call(query: str) -> tuple[float, str]:
    payload = {
        "model": "generator",
        "messages": [
            {"role": "system", "content": LONG_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "max_tokens": 30,
        "stream": False,
    }
    t0 = time.perf_counter()
    resp = httpx.post(GENERATOR_URL, json=payload, timeout=30.0)
    t1 = time.perf_counter()
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return (t1 - t0) * 1000, text


def main():
    print("Request 1 (prefix CHƯA có trong cache, phải tính KV từ đầu):")
    lat1, out1 = call(QUERIES[0])
    print(f"  Latency: {lat1:.1f} ms | Output: {out1!r}\n")

    print("Request 2 (CÙNG prefix hệ thống, câu hỏi khác -> nếu KV cache hoạt "
          "động, latency phải THẤP HƠN vì prefix đã có sẵn trong radix cache):")
    lat2, out2 = call(QUERIES[1])
    print(f"  Latency: {lat2:.1f} ms | Output: {out2!r}\n")

    print("=" * 60)
    diff_pct = (lat1 - lat2) / lat1 * 100 if lat1 else 0
    print(f"Chênh lệch: {diff_pct:.1f}% {'(request 2 nhanh hơn -> KV cache PASS)' if diff_pct > 0 else '(KHÔNG thấy cải thiện -> kiểm tra lại --schedule-policy lpm)'}")
    print("\nLưu ý: chênh lệch có thể không lớn với prompt ngắn/model nhỏ (3B) — "
          "hiệu quả KV cache rõ hơn khi prefix càng dài & nhiều request đồng thời "
          "dùng chung prefix (nhiều khách hỏi cùng lúc, cùng menu/system prompt).")


if __name__ == "__main__":
    main()