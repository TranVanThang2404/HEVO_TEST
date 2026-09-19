# -*- coding: utf-8 -*-
"""
Benchmark ĐÚNG cho Router THẬT của bạn (router_agent_fast.py):
gọi trực tiếp classify_intent() trong process (KHÔNG qua HTTP/SGLang,
vì Router thật chạy in-process bằng log-likelihood scoring trên
Qwen2.5-0.5B-Instruct gốc, không phải model đã fine-tune).

Đặt file này CÙNG THƯ MỤC với router_agent_fast.py (D:\HEVO\A2) rồi chạy:
    python3 benchmark_router_classify_v2.py
"""
import statistics
import json as json_lib

import router_agent_fast as router  # import sẽ tự load model + warm-up baseline

TEST_CASES = [
    ("Cho anh 1 ly cà phê sữa đá", "order"),
    ("Tính tiền giúp em", "order"),
    ("Gọi thêm 1 bánh tiramisu", "order"),
    ("I'd like to order a large iced latte", "order"),
    ("Có gì ngon không em?", "consultant"),
    ("Gợi ý cho tôi món nào mát mát mùa hè", "consultant"),
    ("What do you recommend for a sweet tooth?", "consultant"),
    ("Wifi tên gì vậy em?", "faq"),
    ("Mấy giờ đóng cửa?", "faq"),
    ("Quán có chỗ gửi xe không?", "faq"),
    ("What's your address?", "faq"),
    ("Ừm...", "ignore"),
    ("Hello", "ignore"),
    ("haha", "ignore"),
    ("Có gì ngon rẻ không?", "consultant"),          # hard sample
    ("Món nào bán chạy nhất, cho anh 1 ly", "order"),  # hard sample
]
HARD_TEXTS = {"Có gì ngon rẻ không?", "Món nào bán chạy nhất, cho anh 1 ly"}


def main():
    latencies = []
    n_correct = 0
    n_json_valid = 0
    hard_total = 0
    hard_correct = 0
    rows = []

    print("\n--- WARM-UP (đã chạy lúc import module) xong, bắt đầu đo ---\n")

    for text, expected in TEST_CASES:
        result, latency_ms = router.classify_intent(text)
        predicted = result.get("action")
        json_str = json_lib.dumps(result, ensure_ascii=False)
        try:
            reparsed = json_lib.loads(json_str)
            json_valid = reparsed.get("action") in router.LABELS
        except Exception:
            json_valid = False

        correct = predicted == expected
        latencies.append(latency_ms)
        n_correct += int(correct)
        n_json_valid += int(json_valid)
        if text in HARD_TEXTS:
            hard_total += 1
            hard_correct += int(correct)

        rows.append((text, expected, predicted, json_valid, correct, latency_ms))

    print("=== CHI TIẾT TỪNG CÂU ===")
    for text, expected, predicted, json_valid, correct, ms in rows:
        status = "OK" if correct else "SAI"
        print(f"[{status}] '{text}' -> expected={expected} predicted={predicted} "
              f"json_valid={json_valid} latency={ms:.2f}ms")

    n = len(TEST_CASES)
    print("\n=== TỔNG KẾT (đối chiếu tiêu chí A1) ===")
    print(f"Accuracy: {n_correct}/{n} = {n_correct/n*100:.1f}%  (yêu cầu >= 92%)")
    if hard_total:
        print(f"Hard-sample accuracy: {hard_correct}/{hard_total} = "
              f"{hard_correct/hard_total*100:.1f}%  (yêu cầu >= 75%)")
    print(f"JSON hợp lệ: {n_json_valid}/{n} = {n_json_valid/n*100:.1f}%  (yêu cầu 100%)")
    print(f"Latency avg: {statistics.mean(latencies):.2f} ms")
    print(f"Latency P50: {statistics.median(latencies):.2f} ms")
    srt = sorted(latencies)
    p95 = srt[int(len(srt) * 0.95) - 1] if len(srt) >= 20 else srt[-1]
    print(f"Latency P95 (xấp xỉ, n nhỏ): {p95:.2f} ms")
    print(f"Latency min/max: {min(latencies):.2f} / {max(latencies):.2f} ms")
    print("Ngưỡng: <= 200ms = Đạt, <= 100ms = Xuất sắc (RTX 3060; máy bạn RTX 4050 6GB)")
    print("\nLƯU Ý: model là Qwen2.5-0.5B-Instruct GỐC (chưa fine-tune SFT) + "
          "log-likelihood scoring có calibration -> accuracy đo được ở đây phản ánh "
          "khả năng zero-shot, KHÔNG phải kết quả của quy trình fine-tune theo đúng A1.3.")


if __name__ == "__main__":
    main()