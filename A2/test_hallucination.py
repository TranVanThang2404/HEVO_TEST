"""
B2 — Test Zero Hallucination: Generator (qua FAQAgent/OrderAgent/ConsultantAgent
thật, có RAG context từ Neo4j) có trả lời ĐÚNG theo ground truth (faq.csv/menu.csv)
hay bịa thông tin không.

Gọi THẲNG agent.handle() (bỏ qua Router) để cô lập đúng vấn đề: Generator có
bám context RAG hay không — không lẫn với lỗi phân loại intent của Router.

Chạy:
    python3 test_hallucination.py
"""
from __future__ import annotations

import asyncio
import re

from orchestrator import MultiAgentOrchestrator, MockRouterClient
from request_queue import RequestQueue
from session_store import SessionStore

# ---- Ground truth trích từ faq.csv (1 câu đại diện / category) ----
FAQ_CASES = [
    ("Wifi quán tên gì, mật khẩu là gì?",
     ["Highlands_FreeWifi", "highlands123"]),
    ("Quán mở cửa mấy giờ, đóng cửa mấy giờ?",
     ["6:30", "22:00"]),
    ("Cho hỏi địa chỉ chi nhánh gần đây nhất ở đâu?",
     ["app", "fanpage"]),
    ("Quán có chỗ gửi xe không, có mất phí không?",
     ["miễn phí"]),
    ("Quán nhận thanh toán bằng hình thức nào? Có nhận momo không?",
     ["momo"]),  # kiểm tra tối thiểu 1 hình thức đúng
    ("Có thể đặt bàn trước không?",
     ["hotline", "app"]),
    ("Highlands Coffee có giao hàng tận nơi không?",
     ["grabfood", "shopeefood"]),
    ("Làm sao để đăng ký thành viên tích điểm?",
     ["app", "số điện thoại"]),
    ("Món nào có chứa các loại hạt hoặc gây dị ứng phổ biến?",
     ["hạt"]),
    ("Hiện tại quán có chương trình khuyến mãi gì không?",
     ["app", "fanpage"]),
    ("Quán có phòng máy lạnh, có wifi mạnh để làm việc không?",
     ["máy lạnh", "wifi"]),
]

# ---- Ground truth trích từ menu.csv (giá chính xác) ----
MENU_CASES = [
    ("Cho anh 1 ly cà phê sữa đá size L", "48"),
    ("Cho anh 1 ly trà đào cam sả size M", "49"),
    ("Cho anh 1 phần bánh tiramisu", "36"),
    ("Cho anh 1 ly bạc xỉu size S", "37"),
    ("Cho anh 1 ly freeze matcha size L", "58"),
]

# ---- Câu hỏi KHÔNG có trong dữ liệu (test xem có bịa không) ----
ADVERSARIAL_CASES = [
    "Cho anh 1 ly trà sữa trân châu đường đen size L",  # món không có trong menu.csv
    "Cà phê đen đá size XL giá bao nhiêu?",              # size XL không tồn tại (chỉ S/M/L)
    "Quán có phục vụ heo quay quay không?",              # câu hỏi vô nghĩa, không có trong FAQ
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


async def run_faq_cases(faq_agent) -> tuple[int, int]:
    print("=" * 70)
    print("1. FAQ — kiểm tra bám context (ground truth từ faq.csv)")
    print("=" * 70)
    n_pass = 0
    for question, expected_keywords in FAQ_CASES:
        result = await faq_agent.handle(question, summary="", history=[])
        reply = _norm(result["reply"])
        retrieved = _norm(" ".join(result["meta"]["retrieved"]))

        retrieval_ok = any(_norm(k) in retrieved for k in expected_keywords)
        generation_ok = all(_norm(k) in reply for k in expected_keywords)

        status = "PASS" if generation_ok else "FAIL"
        if generation_ok:
            n_pass += 1
        print(f"\n  [{status}] '{question}'")
        print(f"    Retrieval có chứa keyword: {retrieval_ok}")
        print(f"    Reply: {result['reply'][:200]}")
        if not generation_ok:
            missing = [k for k in expected_keywords if _norm(k) not in reply]
            print(f"    -> THIẾU/SAI: {missing}")
    return n_pass, len(FAQ_CASES)


async def run_menu_cases(order_agent) -> tuple[int, int]:
    print("\n" + "=" * 70)
    print("2. MENU/GIÁ — kiểm tra đúng giá thật (ground truth từ menu.csv)")
    print("=" * 70)
    n_pass = 0
    for question, expected_price_prefix in MENU_CASES:
        result = await order_agent.handle(question, summary="", history=[])
        reply = result["reply"]
        matched_item = result["meta"].get("matched_item")
        price_ok = expected_price_prefix in reply.replace(".", "").replace(",", "")

        status = "PASS" if (matched_item and price_ok) else "FAIL"
        if status == "PASS":
            n_pass += 1
        print(f"\n  [{status}] '{question}'")
        print(f"    Matched item: {matched_item}")
        print(f"    Reply: {reply[:200]}")
        if status == "FAIL":
            print(f"    -> Kỳ vọng giá chứa '{expected_price_prefix}...', matched_item={matched_item}")
    return n_pass, len(MENU_CASES)


async def run_adversarial_cases(order_agent, faq_agent) -> None:
    print("\n" + "=" * 70)
    print("3. ADVERSARIAL — câu hỏi KHÔNG có trong dữ liệu (đọc thủ công, xem có bịa giá/thông tin không)")
    print("=" * 70)
    for question in ADVERSARIAL_CASES:
        agent = order_agent if any(k in question.lower() for k in ["cho anh", "size"]) else faq_agent
        result = await agent.handle(question, summary="", history=[])
        print(f"\n  '{question}'")
        print(f"    matched_item: {result['meta'].get('matched_item', 'N/A')}")
        print(f"    Reply: {result['reply']}")
        print("    -> Kiểm tra thủ công: reply có bịa giá/thông tin cụ thể cho món/thứ KHÔNG tồn tại không?")


async def main():
    orchestrator = MultiAgentOrchestrator(MockRouterClient(), SessionStore(), RequestQueue())
    faq_agent = orchestrator.agents["faq"]
    order_agent = orchestrator.agents["order"]

    faq_pass, faq_total = await run_faq_cases(faq_agent)
    menu_pass, menu_total = await run_menu_cases(order_agent)
    await run_adversarial_cases(order_agent, faq_agent)

    print("\n" + "=" * 70)
    print(f"TỔNG KẾT: FAQ {faq_pass}/{faq_total} PASS | Menu/Giá {menu_pass}/{menu_total} PASS")
    print("Adversarial: cần xem thủ công phần in ở trên (không có ground truth để auto-check).")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())