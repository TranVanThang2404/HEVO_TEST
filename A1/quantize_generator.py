"""
B2.1 — Tải + quantize AWQ cho Generator model (Qwen2.5-3B-Instruct).
Dùng model nhỏ hơn spec gốc (7B) do máy chỉ có 6GB VRAM thật (RTX 4050), không đủ
chạy dual-model 7B+1.5B đồng thời như giả định RTX 3060 12GB của đề bài.
Ghi rõ trong báo cáo: thay thế kiến trúc do giới hạn hardware, tỷ lệ VRAM budget
giữ tương tự tinh thần đề bài (Router nhỏ, Generator lớn hơn nhưng vẫn vừa VRAM).

Chạy (trong WSL, llama-env, có internet để tải model từ HuggingFace):
    python3 quantize_generator.py --out_dir generator_awq
"""
import argparse
import os

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def main(out_dir: str):
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"[Generator AWQ] Đang tải {MODEL_ID} từ HuggingFace (lần đầu sẽ mất vài phút)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoAWQForCausalLM.from_pretrained(MODEL_ID, safetensors=True)

    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    print(f"[Generator AWQ] Đang quantize với config: {quant_config} ...")

    base_samples = [
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
        "Bạc xỉu với cà phê sữa đá khác nhau chỗ nào vậy em?",
        "Cho hỏi menu có món nào decaf không?",
        "Anh muốn đặt giao hàng tận nơi, phí ship bao nhiêu?",
        "Trời nóng vậy uống gì cho mát mà không quá ngọt?",
        "Chi nhánh gần Landmark 81 có mở cửa Chủ nhật không?",
        "Tổng đơn của em bao nhiêu tiền vậy ạ?",
    ]
    calib_data = [f"{a} {b}" for a, b in zip(base_samples, base_samples[::-1])] * 4

    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_data,
        max_calib_samples=len(calib_data),
        max_calib_seq_len=64,
    )

    os.makedirs(out_dir, exist_ok=True)
    model.save_quantized(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[Generator AWQ] Đã lưu vào: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="generator_awq")
    args = parser.parse_args()
    main(args.out_dir)