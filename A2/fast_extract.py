# -*- coding: utf-8 -*-
"""
Fast-path extraction cho C2.2 — loại bỏ từ đệm lịch sự/ngữ cảnh khỏi câu
hỏi bằng regex (rule-based, <1ms), KHÔNG gọi LLM.

LÝ DO TÁCH RIÊNG (ghi chú kiến trúc, đưa vào báo cáo):
Benchmark thật đầu tiên dùng intent_extractor.py (SLM generate) cho MỌI câu
kể cả khi tra cache, kết quả: hit-rate 53.1% (< 60% yêu cầu) và latency
cache-HIT median 551.8ms (>> 100ms yêu cầu). Nguyên nhân gốc: SLM 0.5B
zero-shot trích xuất KHÔNG NHẤT QUÁN giữa các biến thể cùng 1 câu hỏi (vd
biến thể có hậu tố "(hỏi giúp bạn mình)" bị model nuốt mất nội dung câu
hỏi thật, trả về action="hỏi giúp bạn mình") — vừa hại hit-rate (embedding
2 biến thể khác nhau => similarity thấp => miss), vừa hại latency (luôn
phải chờ SLM generate dù đã có trong cache).

Fix kiến trúc: tách {Hành động} dùng làm cache-key thành 1 bước rule-based
nhanh, xác định (chạy MỌI request, kể cả hit). SLM (intent_extractor.py)
CHỈ chạy khi cache MISS thật sự — lúc đó hệ thống vốn đã phải gọi Agent
thật nên thêm vài trăm ms không ảnh hưởng SLA của nhánh hit. Đây là cách
thiết kế hợp lý cho production (không chạy LLM cho input đã từng thấy).

Các mẫu đệm bên dưới là các cụm lịch sự/mở đầu-kết câu PHỔ BIẾN trong giao
tiếp F&B tiếng Việt (không chỉ khớp riêng bộ test faq.csv), tổng quát hoá
từ các ví dụ thật trong dữ liệu công ty cung cấp.
"""
import re

_PREFIX_PATTERNS = [
    r"^(cho\s+(em|anh|chị|tôi|mình)\s+hỏi\s*[:,]?\s*)",
    r"^(anh\s*/?\s*chị\s+ơi,?\s*)",
    r"^(excuse me,?\s*)",
    r"^(mình muốn biết\s*)",
    r"^(xin hỏi\s*)",
    r"^(xin (cho\s+)?(hỏi|biết)\s*)",
    r"^(không biết\s*)",
    r"^(làm ơn cho hỏi\s*)",
    r"^(cho hỏi\s*)",
]

_SUFFIX_PATTERNS = [
    r"(\s*vậy ạ\??\s*)$",
    r"(\s*cảm ơn (shop|ạ|nhé)\.?\s*)$",
    r"(\s*\(hỏi giúp bạn mình\)\s*)$",
    r"(\s*giúp (em|mình|tôi) với\.?\s*)$",
    r"(\s*nhé\.?\s*)$",
    r"(\s*ạ\??\s*)$",
]

_CONTEXT_RE = re.compile(
    r"(\d+\s*(giờ|h|phút|ngày|tuần|tháng|người|ly|phần|đô|k)\b|"
    r"ngày mai|hôm nay|tối nay|sáng mai|sáng nay|chiều nay|cuối tuần|"
    r"sinh nhật)",
    re.IGNORECASE,
)


def strip_wrappers(text: str) -> str:
    """Loại bỏ từ đệm mở đầu/kết thúc, giữ nguyên nội dung câu hỏi cốt lõi.
    Lặp tối đa 3 vòng vì có thể có nhiều lớp đệm lồng nhau (vd vừa có tiền
    tố vừa có hậu tố trong cùng 1 câu)."""
    out = (text or "").strip()
    for _ in range(3):
        prev = out
        for p in _PREFIX_PATTERNS:
            out = re.sub(p, "", out, flags=re.IGNORECASE).strip()
        for p in _SUFFIX_PATTERNS:
            out = re.sub(p, "", out, flags=re.IGNORECASE).strip()
        if out == prev:
            break
    out = out.strip(" ,.")
    return out if out else text.strip()


def extract_context_fast(text: str) -> str:
    spans = [m.group(0) for m in _CONTEXT_RE.finditer(text or "")]
    return ", ".join(dict.fromkeys(spans))  # loại trùng, giữ thứ tự


def guess_subject_fast(text: str) -> str:
    t = (text or "").lower()
    for token in ("anh", "chị", "em", "mình", "tôi"):
        if re.search(rf"\b{token}\b", t):
            return token
    return "khách"


if __name__ == "__main__":
    TEST = [
        "Cho em hỏi wifi quán tên gì, mật khẩu là gì?",
        "Xin hỏi quán mở cửa mấy giờ, đóng cửa mấy giờ?",
        "Quán có chỗ gửi xe không, có mất phí không? (hỏi giúp bạn mình)",
        "Cho anh đặt bàn tiệc sinh nhật vào ngày mai lúc 7h em nhé",
        "Không biết quán có chỗ gửi xe không, có mất phí không? vậy ạ?",
    ]
    for q in TEST:
        print(f"IN : {q}")
        print(f"  action  = '{strip_wrappers(q)}'")
        print(f"  context = '{extract_context_fast(q)}'")
        print(f"  subject = '{guess_subject_fast(q)}'")
        print()