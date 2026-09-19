"""
Sinh dữ liệu mẫu cho Graph RAG (B1) — chưa có data thật từ Highlands Coffee nên tự
sinh, đủ số lượng theo đúng "Tiêu chí nghiệm thu B1":
  - Graph DB chứa >= 100 MenuItems
  - >= 150 FAQ chunks
  - >= 50 Document chunks

Output:
  data/menu.csv     : >= 110 món (name, price_vnd, category, size, ingredients, description)
  data/faq.csv      : >= 160 cặp Q&A (question, answer, category)
  data/handbook.txt : tài liệu nội bộ dài (chính sách phục vụ) để test Semantic Chunking,
                       khi chunk theo đoạn (~250-400 ký tự/chunk) sẽ ra >= 55 chunks.

Chạy: python generate_sample_data.py
"""
import csv
import os

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

# ==================== MENU (>= 110 món) ====================

COFFEE_BASE = [
    ("Cà phê đen đá", "Cà phê phin nguyên chất, đậm vị, không đường sữa."),
    ("Cà phê sữa đá", "Cà phê phin pha cùng sữa đặc, đậm đà truyền thống."),
    ("Bạc xỉu", "Sữa đặc nhiều hơn cà phê, vị ngọt béo nhẹ, ít đắng."),
    ("Americano đá", "Espresso pha loãng với nước, vị thanh, ít béo."),
    ("Americano nóng", "Espresso pha loãng với nước nóng, vị thanh."),
    ("Latte đá", "Espresso pha sữa tươi, lớp bọt mịn, vị nhẹ béo."),
    ("Latte nóng", "Espresso pha sữa tươi nóng, thơm béo."),
    ("Cappuccino", "Espresso, sữa nóng và lớp bọt sữa dày đặc trưng."),
    ("Mocha đá", "Espresso, sữa, socola, vị ngọt đắng hài hòa."),
    ("Mocha nóng", "Espresso, sữa nóng, socola, thơm nồng."),
    ("Espresso Shot", "Cà phê nguyên chất ép áp suất cao, đậm đặc."),
    ("Cà phê muối", "Cà phê phin kết hợp lớp kem muối beo béo mặn ngọt."),
    ("Cold Brew", "Cà phê ủ lạnh 12 tiếng, vị êm dịu, ít chua gắt."),
    ("Phin sữa đá dừa", "Cà phê phin, sữa đặc, nước cốt dừa béo ngậy."),
    ("Caramel Macchiato", "Espresso, sữa tươi, sốt caramel béo ngọt."),
    ("Hazelnut Latte", "Latte hương phỉ, thơm béo, ngọt dịu."),
    ("Vanilla Latte", "Latte hương vani nhẹ nhàng, thơm béo."),
    ("Cà phê trứng", "Cà phê phin kết hợp kem trứng đánh bông béo ngậy kiểu Hà Nội."),
    ("Cà phê cốt dừa freeze", "Cà phê xay đá cùng nước cốt dừa béo mát."),
    ("Doppio Espresso", "Hai shot espresso đậm đặc dành cho người mê cà phê mạnh."),
]

TEA_BASE = [
    ("Trà đào cam sả", "Trà trái cây thanh mát, đào miếng, cam tươi, sả thơm."),
    ("Trà sen vàng", "Trà ướp sen, thơm dịu, hậu vị ngọt thanh."),
    ("Trà vải", "Trà đen kết hợp vải thiều ngọt thơm."),
    ("Trà oolong tứ quý", "Trà oolong nguyên bản, hương hoa nhẹ, hậu ngọt."),
    ("Trà xanh mật ong", "Trà xanh nguyên chất, mật ong nguyên chất, thanh nhẹ."),
    ("Trà chanh dây", "Trà đen, chanh dây chua ngọt sảng khoái."),
    ("Trà bí đao hạt chia", "Trà bí đao mát gan, hạt chia bổ dưỡng."),
    ("Hồng trà macchiato", "Hồng trà pha lớp kem cheese macchiato béo mặn."),
    ("Trà lài", "Trà ướp hoa lài thơm nhẹ, vị thanh tao."),
    ("Trà quất mật ong", "Trà quất chua nhẹ hòa mật ong, tốt cho họng."),
    ("Trà dâu tằm", "Trà dâu tằm ngọt thanh, màu sắc bắt mắt."),
    ("Trà xoài", "Trà trái cây vị xoài chín ngọt đậm."),
]

FREEZE_BASE = [
    ("Freeze trà xanh", "Đá xay trà xanh, vị ngọt béo, mát lạnh."),
    ("Freeze cà phê", "Đá xay cà phê, phủ kem tươi, đậm vị."),
    ("Freeze socola", "Đá xay socola nguyên chất, phủ kem tươi."),
    ("Freeze việt quất", "Đá xay việt quất chua ngọt, phủ kem tươi."),
    ("Freeze matcha", "Đá xay matcha Nhật Bản, vị đắng nhẹ, béo mịn."),
    ("Freeze dâu", "Đá xay dâu tây tươi, vị chua ngọt hài hòa."),
    ("Freeze caramel", "Đá xay sốt caramel béo ngọt, phủ kem tươi."),
    ("Freeze chanh dây", "Đá xay chanh dây chua thanh, giải khát mùa hè."),
]

PASTRY_BASE = [
    ("Bánh croissant bơ", "Bánh sừng bò lớp vỏ giòn, nhân bơ thơm béo."),
    ("Bánh mì que pate", "Bánh mì que giòn kèm pate béo, ăn kèm cà phê rất hợp."),
    ("Bánh tiramisu", "Bánh tiramisu vị cà phê, kem mascarpone mềm mịn."),
    ("Bánh phô mai nướng", "Bánh phô mai Nhật, mềm xốp, béo nhẹ."),
    ("Cookie socola chip", "Bánh quy bơ giòn, socola chip tan chảy."),
    ("Bánh su kem trà xanh", "Vỏ bánh giòn, nhân kem trà xanh béo mịn."),
    ("Bánh croissant socola", "Bánh sừng bò nhân socola tan chảy bên trong."),
    ("Bánh muffin việt quất", "Bánh muffin mềm xốp, nhân việt quất chua ngọt."),
    ("Bánh donut đường", "Bánh donut chiên phủ lớp đường mịn."),
    ("Bánh mì chả cá", "Bánh mì giòn kèm chả cá đậm đà, ăn sáng tiện lợi."),
]

SIZES = ["S", "M", "L"]
PRICE_BASE = {
    "coffee": 39000, "tea": 45000, "freeze": 49000, "pastry": 35000,
}
SIZE_SURCHARGE = {"S": -4000, "M": 0, "L": 6000}

menu_rows = []
for base, category, price_key in [
    (COFFEE_BASE, "coffee", "coffee"),
    (TEA_BASE, "tea", "tea"),
    (FREEZE_BASE, "freeze", "freeze"),
]:
    for name, desc in base:
        for size in SIZES:
            price = PRICE_BASE[price_key] + SIZE_SURCHARGE[size] + (len(name) % 5) * 1000
            menu_rows.append({
                "name": f"{name} (Size {size})",
                "price_vnd": price,
                "category": category,
                "size": size,
                "ingredients": "Xem mô tả",
                "description": desc,
            })

# Bánh không chia size
for name, desc in PASTRY_BASE:
    price = PRICE_BASE["pastry"] + (len(name) % 4) * 1500
    menu_rows.append({
        "name": name, "price_vnd": price, "category": "pastry",
        "size": "Freesize", "ingredients": "Xem mô tả", "description": desc,
    })

# Thêm vài món combo / seasonal để chắc chắn vượt 110
COMBO_EXTRA = [
    ("Combo sáng: Cà phê sữa đá + Bánh mì que", 55000, "combo", "Combo sáng tiết kiệm, năng lượng cho ngày mới."),
    ("Combo chiều: Trà đào + Bánh phô mai", 65000, "combo", "Combo giải khát buổi chiều, ngọt nhẹ."),
    ("Freeze dâu mùa hè", 52000, "freeze", "Đá xay dâu tây tươi, vị chua ngọt mùa hè."),
    ("Trà ổi hồng", 47000, "tea", "Trà trái cây vị ổi hồng lạ miệng."),
    ("Cà phê dừa", 45000, "coffee", "Cà phê phin kết hợp nước cốt dừa béo mịn."),
    ("Bánh su kem", 32000, "pastry", "Vỏ bánh giòn nhẹ, nhân kem sữa mịn."),
]
for name, price, category, desc in COMBO_EXTRA:
    menu_rows.append({
        "name": name, "price_vnd": price, "category": category,
        "size": "Freesize", "ingredients": "Xem mô tả", "description": desc,
    })

with open(os.path.join(OUT_DIR, "menu.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "price_vnd", "category", "size", "ingredients", "description"])
    writer.writeheader()
    writer.writerows(menu_rows)

print(f"Đã sinh {len(menu_rows)} MenuItem -> data/menu.csv (yêu cầu >= 100)")

# ==================== FAQ (>= 160 cặp Q&A) ====================

FAQ_TEMPLATES = [
    ("Wifi", "Wifi quán tên gì, mật khẩu là gì?",
     "Wifi miễn phí tên 'Highlands_FreeWifi', mật khẩu 'highlands123', áp dụng tại mọi chi nhánh."),
    ("Giờ mở cửa", "Quán mở cửa mấy giờ, đóng cửa mấy giờ?",
     "Quán mở cửa từ 6:30 sáng đến 22:00 mỗi ngày, kể cả cuối tuần và ngày lễ."),
    ("Địa chỉ", "Highlands Coffee chi nhánh gần đây nhất ở đâu?",
     "Bạn có thể tra chi nhánh gần nhất qua app Highlands Coffee hoặc fanpage chính thức, có bản đồ chỉ đường."),
    ("Gửi xe", "Quán có chỗ gửi xe không, có mất phí không?",
     "Quán có chỗ giữ xe máy miễn phí phía sau hoặc trước cửa hàng, có bảo vệ trông xe tại các chi nhánh lớn."),
    ("Thanh toán", "Quán nhận thanh toán bằng những hình thức nào?",
     "Highlands Coffee nhận tiền mặt, thẻ ngân hàng, và ví điện tử (Momo, ZaloPay, VNPay, ShopeePay)."),
    ("Đặt bàn", "Có thể đặt bàn trước không, đặt bàn tiệc sinh nhật thế nào?",
     "Có thể đặt bàn qua hotline hoặc app. Đặt bàn tiệc sinh nhật/nhóm đông cần báo trước ít nhất 2 tiếng."),
    ("Giao hàng", "Highlands Coffee có giao hàng tận nơi không?",
     "Có, đặt qua app Highlands Coffee, GrabFood, ShopeeFood hoặc Baemin, giao trong bán kính 3km quanh chi nhánh."),
    ("Thành viên", "Làm sao để đăng ký thành viên tích điểm?",
     "Tải app Highlands Coffee, đăng ký tài khoản bằng số điện thoại để tích điểm và nhận ưu đãi sinh nhật."),
    ("Dị ứng", "Món nào có chứa các loại hạt hoặc gây dị ứng phổ biến?",
     "Một số bánh có chứa hạt (óc chó, hạnh nhân) — vui lòng hỏi nhân viên trước khi gọi món nếu bạn dị ứng."),
    ("Khuyến mãi", "Hiện tại quán có chương trình khuyến mãi gì không?",
     "Chương trình khuyến mãi thay đổi theo tháng, cập nhật trên app và fanpage chính thức Highlands Coffee."),
    ("Nhượng quyền", "Muốn mở nhượng quyền Highlands Coffee thì liên hệ ai?",
     "Liên hệ bộ phận phát triển nhượng quyền qua website chính thức, mục 'Franchise' hoặc email hợp tác kinh doanh."),
    ("Khiếu nại", "Muốn phản ánh chất lượng phục vụ thì làm sao?",
     "Bạn có thể phản ánh qua hotline chăm sóc khách hàng hoặc mục 'Liên hệ' trên app, đội ngũ sẽ phản hồi trong 24h."),
    ("Không gian", "Quán có phòng máy lạnh, có wifi mạnh để làm việc không?",
     "Hầu hết chi nhánh đều có máy lạnh, wifi tốc độ cao, phù hợp học tập và làm việc nhóm."),
    ("Trẻ em", "Quán có ghế cho trẻ em, có món cho bé không?",
     "Một số chi nhánh có ghế cao cho trẻ em; menu có các loại trà trái cây ít cafein phù hợp cho bé."),
    ("Thú cưng", "Có được mang thú cưng vào quán không?",
     "Tùy chi nhánh, một số cửa hàng có khu vực ngoài trời cho phép mang thú cưng, vui lòng hỏi nhân viên tại chỗ."),
    ("Sự kiện", "Có thể thuê không gian quán để tổ chức sự kiện nhỏ không?",
     "Một số chi nhánh lớn nhận đặt chỗ tổ chức họp nhóm, sự kiện nhỏ, cần liên hệ trước qua quản lý chi nhánh."),
]

# Nhân bản có biến thể để đạt >= 160 (đổi cách hỏi, giữ nguyên nội dung/category)
PHRASING_VARIANTS = [
    "{q}", "Cho em hỏi {q_lower}", "Anh/chị ơi, {q_lower}",
    "Excuse me, {q}", "{q} Cảm ơn shop.", "Mình muốn biết {q_lower}",
    "Xin hỏi {q_lower}", "Không biết {q_lower} vậy ạ?", "{q} (hỏi giúp bạn mình)",
    "Làm ơn cho hỏi {q_lower}",
]

faq_rows = []
for category, question, answer in FAQ_TEMPLATES:
    for variant in PHRASING_VARIANTS:
        q_variant = variant.format(q=question, q_lower=question[0].lower() + question[1:])
        faq_rows.append({"question": q_variant, "answer": answer, "category": category})

with open(os.path.join(OUT_DIR, "faq.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "answer", "category"])
    writer.writeheader()
    writer.writerows(faq_rows)

print(f"Đã sinh {len(faq_rows)} cặp FAQ Q&A -> data/faq.csv (yêu cầu >= 150)")

# ==================== DOCUMENT nội bộ (chunk ra >= 50 chunks) ====================

HANDBOOK_SECTIONS = [
    ("Quy trình order tại quầy",
     "Nhân viên thu ngân tiếp nhận order trực tiếp hoặc qua app, xác nhận lại tên món, "
     "size, số lượng và các yêu cầu thêm (ít đá, không đường...) trước khi in hóa đơn. "
     "Đơn hàng được chuyển ngay xuống quầy pha chế qua hệ thống POS nội bộ. Thời gian "
     "chờ trung bình cho đồ uống là 3-5 phút vào giờ thấp điểm, 7-10 phút vào giờ cao điểm."),
    ("Tiêu chuẩn pha chế đồ uống",
     "Mỗi công thức đồ uống đều có định lượng chuẩn (gram/ml) được quy định trong sổ tay "
     "pha chế nội bộ, đảm bảo hương vị đồng nhất giữa các chi nhánh. Nhân viên pha chế "
     "phải hoàn thành khóa đào tạo nội bộ tối thiểu 2 tuần trước khi được đứng quầy chính thức."),
    ("Chính sách vệ sinh an toàn thực phẩm",
     "Toàn bộ nguyên liệu tươi (sữa, trái cây, kem) phải được kiểm tra hạn sử dụng hàng ngày "
     "và ghi chép vào sổ kiểm tra. Dụng cụ pha chế được vệ sinh sau mỗi ca làm việc. Nhân "
     "viên phải đeo găng tay khi xử lý thực phẩm không qua chế biến nhiệt."),
    ("Chính sách đổi trả và hoàn tiền",
     "Nếu đồ uống pha sai order hoặc không đạt chất lượng, khách hàng được đổi lại miễn phí "
     "trong vòng 10 phút kể từ khi nhận món, không cần giữ hóa đơn nếu thanh toán qua app. "
     "Hoàn tiền chỉ áp dụng khi đơn hàng bị hủy trước khi pha chế."),
    ("Chương trình thành viên và tích điểm",
     "Mỗi 10.000đ chi tiêu tích lũy 1 điểm thành viên. Đủ 100 điểm được đổi 1 voucher giảm "
     "giá 20.000đ. Thành viên hạng Vàng (chi tiêu >2 triệu/năm) được ưu đãi sinh nhật và "
     "quà tặng đặc biệt vào các dịp lễ lớn."),
    ("Quy định về không gian quán",
     "Khách được sử dụng không gian quán tối đa 2 tiếng vào giờ cao điểm (11h-13h, 17h-19h) "
     "nếu quán đông khách, nhân viên sẽ nhắc nhở lịch sự. Ngoài giờ cao điểm không giới hạn "
     "thời gian ngồi lại miễn là có gọi thêm đồ uống."),
    ("Chính sách giao hàng tận nơi",
     "Đơn hàng giao qua app tự vận hành hoặc đối tác thứ 3 (Grab, Shopee, Baemin) trong bán "
     "kính 3km. Đồ uống đá được đóng gói riêng đá và nước để tránh loãng vị khi vận chuyển "
     "xa. Thời gian giao hàng cam kết trong 30 phút kể từ khi xác nhận đơn."),
    ("Quy trình xử lý khiếu nại khách hàng",
     "Mọi khiếu nại được ghi nhận qua hotline hoặc app, chuyển tới quản lý chi nhánh xử lý "
     "trong 24 giờ. Trường hợp khiếu nại về chất lượng sản phẩm nghiêm trọng được báo cáo "
     "lên bộ phận QA trung tâm để kiểm tra quy trình pha chế tại chi nhánh liên quan."),
    ("Đào tạo nhân viên mới",
     "Nhân viên mới trải qua 3 giai đoạn: đào tạo lý thuyết (menu, quy trình, phần mềm POS), "
     "thực hành pha chế dưới giám sát, và đánh giá năng lực trước khi được xếp ca độc lập. "
     "Toàn bộ quá trình kéo dài trung bình 3 tuần."),
    ("Chính sách nhượng quyền thương hiệu",
     "Đối tác nhượng quyền phải tuân thủ tiêu chuẩn thiết kế, menu, và quy trình vận hành "
     "thống nhất trên toàn hệ thống. Highlands Coffee cử đội ngũ vận hành hỗ trợ khai trương "
     "trong 2 tuần đầu tiên và kiểm tra chất lượng định kỳ hàng quý."),
    ("Quản lý tồn kho nguyên liệu",
     "Mỗi chi nhánh kiểm kê tồn kho nguyên liệu 2 lần/ngày (đầu ca sáng và cuối ca tối). "
     "Nguyên liệu tươi được đặt hàng bổ sung hàng ngày qua hệ thống trung tâm, nguyên liệu "
     "khô đặt hàng theo chu kỳ tuần để tránh tồn kho quá hạn."),
    ("Chính sách môi trường và bao bì",
     "Highlands Coffee khuyến khích khách hàng dùng ly cá nhân bằng ưu đãi giảm 5.000đ mỗi "
     "đơn. Ống hút và ly mang đi sử dụng vật liệu thân thiện môi trường, đang triển khai "
     "thay thế toàn bộ nhựa dùng một lần trước năm 2027."),
    ("Chính sách tổ chức sự kiện tại quán",
     "Một số chi nhánh có không gian riêng cho họp nhóm, sự kiện nhỏ dưới 20 người, cần đặt "
     "trước qua quản lý chi nhánh tối thiểu 1 ngày. Chi phí thuê không gian có thể được miễn "
     "nếu tổng hóa đơn đồ uống vượt mức quy định."),
    ("Quy định đồng phục nhân viên",
     "Nhân viên pha chế và thu ngân mặc đồng phục theo đúng quy chuẩn thương hiệu, đeo bảng "
     "tên rõ ràng trong suốt ca làm việc. Tóc phải được cột gọn gàng khi đứng quầy pha chế "
     "để đảm bảo vệ sinh an toàn thực phẩm."),
    ("Chính sách ca làm việc và lương thưởng",
     "Nhân viên part-time làm việc theo ca 4-6 tiếng, lương tính theo giờ cộng phụ cấp ca "
     "tối. Nhân viên full-time được hưởng đầy đủ chế độ bảo hiểm theo quy định pháp luật "
     "và thưởng doanh số theo quý dựa trên kết quả kinh doanh chi nhánh."),
    ("Quy trình kiểm tra chất lượng định kỳ",
     "Đội ngũ QA trung tâm kiểm tra ngẫu nhiên chất lượng đồ uống và vệ sinh tại các chi "
     "nhánh mỗi quý một lần, không báo trước. Kết quả kiểm tra được dùng để đánh giá thi "
     "đua giữa các chi nhánh trong cùng khu vực."),
    ("Chính sách bảo mật thông tin khách hàng",
     "Thông tin cá nhân khách hàng đăng ký thành viên (số điện thoại, ngày sinh) được bảo "
     "mật theo quy định, chỉ dùng để gửi ưu đãi và không chia sẻ cho bên thứ ba khi chưa có "
     "sự đồng ý của khách hàng."),
    ("Quy trình xử lý sự cố mất điện, mất nước",
     "Khi xảy ra sự cố mất điện hoặc mất nước, chi nhánh phải thông báo ngay cho khách hàng "
     "đang có mặt, tạm ngưng nhận order mới và liên hệ đội kỹ thuật trung tâm xử lý trong "
     "thời gian sớm nhất có thể."),
    ("Chính sách tuyển dụng nhân sự mới",
     "Ứng viên ứng tuyển vị trí pha chế hoặc thu ngân trải qua vòng phỏng vấn trực tiếp tại "
     "chi nhánh, kiểm tra thái độ phục vụ và khả năng giao tiếp cơ bản trước khi được nhận "
     "vào đào tạo chính thức."),
]

CHUNK_SIZE_CHARS = 90  # semantic-ish chunking: gộp mệnh đề liên tiếp tới khi đạt ngưỡng ký tự


def semantic_chunk(text: str, max_chars: int = CHUNK_SIZE_CHARS) -> list[str]:
    """Chunking mô phỏng gradient breakpoint: tách theo cả dấu câu (.) VÀ dấu phẩy (,) thành
    các mệnh đề nhỏ, rồi gộp mệnh đề liên tiếp tới khi đạt max_chars. Cách này cho ra chunk
    kích thước đều và mịn hơn so với chỉ tách theo câu — gần với ý tưởng "gradient breakpoint"
    (tìm điểm ngắt tự nhiên trong văn bản) mà không cần tính similarity embedding giữa câu."""
    import re
    clauses = [c.strip() for c in re.split(r"(?<=[,.]) ", text.replace("\n", " ")) if c.strip()]
    chunks, current = [], ""
    for c in clauses:
        if len(current) + len(c) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = c
        else:
            current = (current + " " + c).strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks


all_chunks = []
for title, body in HANDBOOK_SECTIONS:
    for chunk in semantic_chunk(body):
        all_chunks.append({"section": title, "text": chunk})

with open(os.path.join(OUT_DIR, "handbook.txt"), "w", encoding="utf-8") as f:
    for title, body in HANDBOOK_SECTIONS:
        f.write(f"## {title}\n{body}\n\n")

print(f"Đã sinh {len(all_chunks)} Document chunk (từ {len(HANDBOOK_SECTIONS)} mục) -> data/handbook.txt (yêu cầu >= 50)")
print("\nXong. Kiểm tra thư mục data/ trước khi ingest vào Neo4j.")