"""
A1.3 (tiếp) — Quantize Router model đã fine-tune, xuất 2 định dạng theo đúng yêu cầu đề bài:

  1. AWQ (INT4) — cho serving SGLang/vLLM trên RTX 3060.
  2. GGUF (Q4_K_M) — cho Edge deployment (Orange Pi / CPU), dùng llama.cpp.

Yêu cầu cài đặt (chạy trong WSL Ubuntu, KHÔNG chạy trên Windows native vì autoawq cần
build CUDA extension, dễ lỗi trên Windows):

    pip install autoawq==0.2.6 transformers==4.44.2 torch==2.4.0 --break-system-packages

Với GGUF, cần clone + build llama.cpp (chỉ cần công cụ convert, không cần build server):

    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    pip install -r requirements.txt --break-system-packages

Chạy (từ thư mục chứa router_finetuned_output/router_merged):
    python quantize_router.py --model_dir router_finetuned_output/router_merged \
        --awq_out router_awq --llamacpp_dir /path/to/llama.cpp --gguf_out router_gguf
"""
import argparse
import json
import os
import subprocess
import sys


def quantize_awq(model_dir: str, out_dir: str):
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"[AWQ] Đang nạp model từ {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoAWQForCausalLM.from_pretrained(model_dir, safetensors=True)

    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    print(f"[AWQ] Đang quantize với config: {quant_config} ...")

    # Calibration data: dùng vài câu tiếng Việt/Anh mẫu thuộc domain router để calibrate
    # tốt hơn so với calibration set mặc định (thường là tiếng Anh tổng quát).
    base_samples = [
        "Cho anh 1 ly cà phê sữa đá size L mang đi nhé.",
        "Có món nào ngọt mà không quá béo không?",
        "Wifi pass là gì vậy em?",
        "Alo alo test mic haha",
        "I'd like a large iced latte please.",
        "What do you recommend for a hot day?",
        "What time do you close today?",
        "Hello, just testing the mic.",
        "Cho tôi hỏi quán còn chỗ ngồi không, đông khách quá.",
        "Tính tiền giúp em với 2 ly trà sữa trân châu.",
        "Gợi ý cho tôi 1 món ít đường, ít calo được không?",
        "Món này giá bao nhiêu và có size nhỏ hơn không?",
        "Can I get a recommendation for something not too sweet?",
        "How much does a medium iced coffee cost here?",
        "Umm... never mind, just checking the connection.",
        "Xin chào, cho hỏi giờ mở cửa của chi nhánh này là mấy giờ?",
    ]
    # Nhân đôi + nối cặp câu lại để mỗi mẫu calib đủ dài hơn (AutoAWQ mặc định cắt
    # calibration text thành đoạn max_calib_seq_len token; câu quá ngắn -> mẫu hợp lệ = 0).
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
    print(f"[AWQ] Đã lưu model AWQ INT4 vào: {out_dir}")


def quantize_gguf(model_dir: str, llamacpp_dir: str, out_dir: str, quant_type: str = "Q4_K_M"):
    os.makedirs(out_dir, exist_ok=True)
    fp16_gguf = os.path.join(out_dir, "router_fp16.gguf")
    final_gguf = os.path.join(out_dir, f"router_{quant_type}.gguf")

    convert_script = os.path.join(llamacpp_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        # Tên script thay đổi giữa các version llama.cpp, thử tên cũ hơn.
        convert_script = os.path.join(llamacpp_dir, "convert-hf-to-gguf.py")

    print(f"[GGUF] Bước 1/2: Convert HF -> GGUF FP16 ...")
    subprocess.run([sys.executable, convert_script, model_dir, "--outfile", fp16_gguf, "--outtype", "f16"], check=True)

    quantize_bin = os.path.join(llamacpp_dir, "llama-quantize")
    if not os.path.exists(quantize_bin):
        quantize_bin = os.path.join(llamacpp_dir, "quantize")
    if not os.path.exists(quantize_bin):
        raise FileNotFoundError(
            f"Không tìm thấy binary quantize trong {llamacpp_dir}. "
            f"Cần build llama.cpp trước: cd {llamacpp_dir} && cmake -B build && cmake --build build --config Release"
        )

    print(f"[GGUF] Bước 2/2: Quantize FP16 -> {quant_type} ...")
    subprocess.run([quantize_bin, fp16_gguf, final_gguf, quant_type], check=True)
    print(f"[GGUF] Đã lưu model GGUF {quant_type} vào: {final_gguf}")

    fp16_size = os.path.getsize(fp16_gguf) / 1e6
    final_size = os.path.getsize(final_gguf) / 1e6
    print(f"[GGUF] Kích thước: FP16={fp16_size:.1f}MB -> {quant_type}={final_size:.1f}MB "
          f"(giảm {(1 - final_size/fp16_size)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Thư mục chứa model đã merge LoRA (router_merged)")
    parser.add_argument("--awq_out", default="router_awq")
    parser.add_argument("--llamacpp_dir", default=None, help="Đường dẫn tới thư mục llama.cpp đã clone+build")
    parser.add_argument("--gguf_out", default="router_gguf")
    parser.add_argument("--gguf_type", default="Q4_K_M")
    parser.add_argument("--skip_awq", action="store_true")
    parser.add_argument("--skip_gguf", action="store_true")
    args = parser.parse_args()

    if not args.skip_awq:
        quantize_awq(args.model_dir, args.awq_out)
    if not args.skip_gguf:
        if not args.llamacpp_dir:
            print("[GGUF] Bỏ qua (chưa truyền --llamacpp_dir).")
        else:
            quantize_gguf(args.model_dir, args.llamacpp_dir, args.gguf_out, args.gguf_type)

    print("\nHoàn tất quantize.")