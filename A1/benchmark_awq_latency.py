"""
A1.3 (cuối) — Benchmark latency của Router model đã AWQ-quantize.
Đo trên chính GPU đang có (RTX 4050 6GB) — dùng để so sánh tương đối, ghi rõ trong
báo cáo là khác RTX 3060 12GB theo đề bài (không có máy đúng spec để test).

Đo 2 chỉ số:
  - TTFT-tương-đương cho task classification (router chỉ sinh 1-3 token nhãn, nên
    "TTFT" ở đây gần như = tổng latency luôn, vẫn đo riêng để đối chiếu format với B2).
  - Latency toàn phần / request (ms), trên N câu query mẫu, sau khi warm-up.

Chạy (trong WSL, llama-env, thư mục D:\\HEVO\\A1):
    python3 benchmark_awq_latency.py --model_dir router_awq --n 30
"""
import argparse
import time

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

TEST_QUERIES = [
    "Cho anh 1 ly cà phê sữa đá size L mang đi nhé.",
    "Có món nào ngọt mà không quá béo không?",
    "Wifi pass là gì vậy em?",
    "Quán mở cửa mấy giờ vậy?",
    "Cho tôi 2 ly trà sữa trân châu ship tới 123 Lê Lợi.",
    "Trà đào cam sả giá bao nhiêu?",
    "Em ơi cho chị hỏi có chỗ đậu xe không?",
    "Tôi muốn đặt bàn cho 4 người lúc 7h tối nay.",
    "Có ưu đãi gì cho sinh viên không shop?",
    "Cho anh 1 phần bánh mì que và 1 ly bạc xỉu.",
] * 3  # 30 câu


def build_prompt(tokenizer, query: str) -> str:
    messages = [
        {"role": "system", "content": "Bạn là router phân loại ý định: order, consultant, faq, hoặc ignore. Chỉ trả về đúng 1 từ nhãn."},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main(model_dir: str, n: int):
    print(f"[Benchmark] Đang nạp model AWQ từ {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoAWQForCausalLM.from_quantized(model_dir, fuse_layers=False, safetensors=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Benchmark] Device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    queries = TEST_QUERIES[:n]

    # Warm-up (loại bỏ chi phí CUDA context / cudnn autotune lần đầu)
    prompt = build_prompt(tokenizer, queries[0])
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    for _ in range(3):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=5, do_sample=False)
    if device == "cuda":
        torch.cuda.synchronize()

    latencies_ms = []
    for q in queries:
        prompt = build_prompt(tokenizer, q)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=5, do_sample=False)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    latencies_ms.sort()
    n_lat = len(latencies_ms)
    p50 = latencies_ms[n_lat // 2]
    p95 = latencies_ms[int(n_lat * 0.95) - 1]
    avg = sum(latencies_ms) / n_lat

    print("\n" + "=" * 60)
    print(f"KẾT QUẢ LATENCY BENCHMARK (n={n_lat}, model=AWQ INT4, device={device})")
    print("=" * 60)
    print(f"  Avg : {avg:.1f} ms")
    print(f"  P50 : {p50:.1f} ms")
    print(f"  P95 : {p95:.1f} ms")
    print(f"  Min : {min(latencies_ms):.1f} ms")
    print(f"  Max : {max(latencies_ms):.1f} ms")
    print("\n  Mục tiêu đề bài: <=200ms (Đạt) / <=100ms (Xuất sắc), đo trên RTX 3060 12GB.")
    print(f"  Máy test hiện tại: {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'} "
          f"(khác RTX 3060 -> ghi chú rõ trong báo cáo).")
    for label, budget in [("Đạt (<=200ms)", 200), ("Xuất sắc (<=100ms)", 100)]:
        status = "PASS" if p50 <= budget else "FAIL"
        print(f"  P50 vs {label}: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="router_awq")
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()
    main(args.model_dir, args.n)