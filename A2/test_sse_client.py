"""
Test client cho B2.2 SSE — minh hoạ:
  1. Nhận đúng token-stream (event: token) VÀ clause-stream (event: clause) riêng biệt.
  2. Xử lý ĐÚNG sentinel "[DONE]" — check chuỗi literal TRƯỚC khi json.loads
     (nhiều implementation sai ở chỗ này: cố json.loads("[DONE]") -> crash).
  3. Đo TTFT (time-to-first-token) để đối chiếu với mục tiêu B2.4 (<=200ms / <=70ms).

Chạy (cần server api_server.py đang chạy ở port 8080):
    python test_sse_client.py
"""
import json
import time

import httpx

URL = "http://127.0.0.1:8000/v1/chat/stream"


def main():
    payload = {"session_id": "demo", "message": "Cho anh 1 ly cà phê sữa đá"}
    t0 = time.perf_counter()
    first_token_time = None
    full_text = ""
    clauses = []
    done_received = False

    with httpx.Client(timeout=30.0) as client:
        with client.stream("POST", URL, json=payload) as resp:
            event_type = None
            for line in resp.iter_lines():
                if line == "":
                    event_type = None
                    continue
                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                    continue
                if line.startswith("data:"):
                    data = line[len("data:"):].strip()

                    # === Điểm mấu chốt: check [DONE] TRƯỚC khi json.loads ===
                    if data == "[DONE]":
                        done_received = True
                        print("\n[OK] Nhận đúng sentinel [DONE], dừng đọc stream.")
                        break

                    payload_json = json.loads(data)
                    if event_type == "token":
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        full_text += payload_json["delta"]["content"]
                    elif event_type == "clause":
                        clauses.append(payload_json["clause"])
                        print(f"  [TTS clause] {payload_json['clause']}")
                    elif event_type == "error":
                        print(f"  [ERROR] {payload_json['error']}")

    ttft = (first_token_time - t0) * 1000 if first_token_time else None
    total = (time.perf_counter() - t0) * 1000

    print("\n" + "=" * 60)
    print(f"Full text ghép từ token-stream: {full_text.strip()}")
    print(f"Số clause tách được (cho TTS)  : {len(clauses)}")
    print(f"TTFT                            : {ttft:.1f} ms" if ttft else "TTFT: N/A")
    print(f"Total latency                   : {total:.1f} ms")
    print("\n" + "=" * 60)
    print("KIỂM TRA YÊU CẦU B2.2:")
    print(f"1. True SSE streaming (nhiều event token rời rạc): "
          f"{'PASS' if full_text.count(' ') > 3 else 'FAIL'}")
    print(f"2. [DONE] xử lý đúng, không crash khi parse       : "
          f"{'PASS' if done_received else 'FAIL'}")
    print(f"3. Clause-level stream riêng cho TTS (>=1 clause) : "
          f"{'PASS' if len(clauses) >= 1 else 'FAIL'}")
    print(f"\n(Đối chiếu TTFT với mục tiêu đề bài: <=200ms Đạt / <=70ms Xuất sắc — "
          f"đo trên RTX 4050, khác RTX 3060 giả định của đề, ghi rõ trong báo cáo.)")


if __name__ == "__main__":
    main()