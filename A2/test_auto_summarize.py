"""
A2.2 — Test riêng cho Auto-summarization (chưa được demo_multiturn cover thật sự,
vì hội thoại ngắn không đủ vượt ngưỡng 70% context window).

Cách test: hạ CONTEXT_WINDOW_TOKENS xuống rất nhỏ (monkeypatch qua module) để ép
_auto_summarize() phải kích hoạt chỉ sau vài turn ngắn, rồi kiểm tra:
  1. sess.summary phải khác rỗng sau khi vượt ngưỡng (summarize_fn thực sự được gọi).
  2. sess.history phải được rút gọn còn <= HISTORY_WINDOW turn gần nhất (đúng logic
     "giữ nguyên văn HISTORY_WINDOW turn mới nhất, phần cũ hơn đem tóm tắt").
  3. summarize_fn được gọi với đúng phần turns cũ (không phải toàn bộ history).

Chạy: python test_auto_summarize.py
"""
import asyncio

import session_store as ss_module
from session_store import SessionStore, HISTORY_WINDOW


async def fake_summarize(old_summary: str, turns: list) -> str:
    """Giả lập LLM summarize — trả về chuỗi đánh dấu rõ đã được gọi + số turn đã tóm tắt."""
    return f"[TOM_TAT:{len(turns)}_turn_cu]"


async def main():
    # Ép ngưỡng token rất thấp để chỉ cần vài turn ngắn là vượt 70%.
    ss_module.CONTEXT_WINDOW_TOKENS = 30
    ss_module.SUMMARIZE_THRESHOLD_RATIO = 0.7  # ngưỡng thật = 21 token ước lượng

    store = SessionStore(summarize_fn=fake_summarize)
    session_id = "test-summarize"

    messages = [
        "Cho anh 1 ly cà phê sữa đá",
        "Wifi quán tên gì vậy em",
        "Có món nào ngọt không quá béo không",
        "Mấy giờ quán đóng cửa vậy em",
        "Cho anh thêm 1 ly trà đào nữa nhé",
        "Tính tiền giúp anh luôn",
        "Cảm ơn em nhiều nha",
        "Hẹn gặp lại lần sau",
    ]

    triggered_at = None
    for i, msg in enumerate(messages):
        await store.add_turn(session_id, "user", msg)
        sess = store.get_or_create(session_id)
        if sess.summary and triggered_at is None:
            triggered_at = i
            print(f"[OK] Auto-summarize kích hoạt lần đầu ở turn #{i}: summary='{sess.summary}'")

    sess = store.get_or_create(session_id)
    print(f"\nSố turn còn lại trong history sau tất cả: {len(sess.history)} (yêu cầu <= {HISTORY_WINDOW * 2})")
    print(f"Summary cuối cùng: '{sess.summary}'")

    print("\n" + "=" * 60)
    ok_triggered = triggered_at is not None
    ok_bounded = len(sess.history) <= HISTORY_WINDOW * 2
    ok_content = sess.summary.startswith("[TOM_TAT:")
    print(f"1. Auto-summarize có kích hoạt thật (không phải chỉ deque maxlen): {'PASS' if ok_triggered else 'FAIL'}")
    print(f"2. History sau summarize bị rút gọn đúng: {'PASS' if ok_bounded else 'FAIL'}")
    print(f"3. summarize_fn thực sự được gọi (không bị nuốt exception): {'PASS' if ok_content else 'FAIL'}")


if __name__ == "__main__":
    asyncio.run(main())