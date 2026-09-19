# MAS-LLM-2026-FINAL — Highlands Coffee Multi-Agent LLM & Graph RAG System

Hệ thống trợ lý ảo quán cà phê Highlands, gồm 3 phần theo đúng cấu trúc đề bài:
- **Phần A** — Multi-Agent Router & Data Pipeline (35%)
- **Phần B** — Graph RAG & LLM Serving (40%)
- **Phần C** — Intelligent Cache & Edge (25%)

Phần cứng thực đo: NVIDIA RTX 4050 Laptop 6GB VRAM (WSL2 Ubuntu trên Windows) —
khác giả định đề bài (RTX 3060 12GB), một số benchmark bị giới hạn bởi VRAM nhỏ
hơn, đã ghi rõ trong từng báo cáo.

## Cấu trúc thư mục

Vị trí file giữ nguyên như hiện tại (không di chuyển, để không phá import
đang chạy được) — cây dưới đây chỉ nhóm lại theo CHỨC NĂNG cho dễ theo dõi.

```
HEVO/
│
├── A1/  — Router: sinh data, fine-tune, quantize
│   ├── generate_router_data.py      # sinh data huấn luyện Router
│   ├── clean_router_dataset.py      # làm sạch data
│   ├── finetune_router.py           # LoRA SFT cho Router
│   ├── quantize_router.py           # quantize AWQ Router
│   ├── quantize_generator.py        # quantize AWQ Generator
│   ├── router_dataset*.jsonl        # data đã sinh + đã làm sạch
│   └── router_finetuned_output/     # eval_report.json, confusion_matrix.png
│       └── router_merged/           # ⚠ checkpoint đã XÁC NHẬN BỊ HỎNG —
│                                     #   KHÔNG dùng, xem báo cáo A để biết lý do
│
├── A2/  — Multi-Agent system (chạy được, Phần A + C)
│   │
│   ├── [Core — Router & Orchestrator, Phần A]
│   │   ├── router_agent_fast.py     # Router (Qwen2.5-0.5B base, log-likelihood scoring)
│   │   ├── orchestrator.py          # điều phối 3 agent theo intent
│   │   ├── agents.py                # Order / Consultant / FAQ agent
│   │   ├── session_store.py         # lưu trạng thái hội thoại multi-turn
│   │   ├── request_queue.py         # hàng đợi request
│   │   └── api_server.py            # FastAPI, SSE streaming
│   │
│   ├── [Kết nối sang Phần B]
│   │   ├── cache_layer.py           # GraphCache, ModelCache, SemanticCache (B2.3)
│   │   └── neo4j_knowledge_base.py  # cầu nối sang B1/hybrid_search.py
│   │
│   ├── [Phần C — Intelligent Cache & Guardrails]
│   │   ├── intent_extractor.py      # C2.1 SLM Intent Extraction (đã fine-tune)
│   │   ├── generate_intent_dataset.py  # sinh data huấn luyện C2.1
│   │   ├── finetune_intent_sft.py   # LoRA SFT cho C2.1
│   │   ├── verify_merged_checkpoint.py # kiểm tra checkpoint C2.1 không hỏng
│   │   ├── evaluate_intent_extractor.py # đo accuracy C2.1 (98.6%)
│   │   ├── intent_sft_lora/         # (gitignore) LoRA adapter đã train
│   │   ├── intent_sft_merged/       # (gitignore) checkpoint C2.1 đã merge
│   │   ├── fast_extract.py          # C2.2 fast-path rule-based (cache-key, <1ms)
│   │   ├── semantic_cache_c2.py     # C2.2 Cache Pipeline & Paraphrase
│   │   ├── guardrails.py            # C3 Production Guardrails (TTS/rate-limit/circuit-breaker/health)
│   │   └── c2_data/                 # faq.csv, menu.csv, intent_dataset.jsonl
│   │
│   └── [Demo & Test]
│       ├── demo_multiturn.py        # Demo video Phần A
│       ├── demo_graphrag_stream.py  # Demo video Phần B
│       ├── demo_part_c.py           # Demo video Phần C (cache + guardrails)
│       ├── test_guardrails.py       # test độc lập C3
│       ├── c2_benchmark.py          # benchmark hit-rate/latency C2.2
│       └── test_*.py, benchmark_*.py  # các test/benchmark khác (A1, B)
│
├── B1/  — Graph RAG Knowledge Base (Phần B)
│   ├── graph_schema.py              # schema Neo4j
│   ├── ingest.py                    # pipeline nạp dữ liệu vào Neo4j
│   ├── embeddings.py                # Qwen3-Embedding-0.6B + reranker
│   ├── hybrid_search.py             # dual-domain search + expansion + rerank
│   ├── data/                        # faq.csv, menu.csv, handbook.txt (dữ liệu nguồn)
│   └── docker-compose.yml           # container Neo4j
│
├── deploy/
│   └── docker-compose.yml           # compose cho toàn hệ thống
│
└── docs/
    ├── BaoCao_KyThuat_A.pdf
    ├── BaoCao_KyThuat_B.pdf
    └── BaoCao_KyThuat_C.pdf
```

**Cách đọc nhanh:** mỗi phần đề bài (A/B/C) có 1 báo cáo PDF trong `docs/` +
1 demo trong `A2/demo_*.py`. `A1/` chỉ chứa script build Router (chạy 1 lần
lúc chuẩn bị), còn `A2/` là hệ thống chạy thật lúc serving. `B1/` độc lập,
được `A2/neo4j_knowledge_base.py` gọi sang khi cần Graph RAG.

## Cách chạy nhanh

### Yêu cầu
- Python 3.11, CUDA GPU (đã test trên RTX 4050 6GB)
- Neo4j (`docker compose -f B1/docker-compose.yml up -d`)
- `pip install -r A2/requirements.txt` (và `B1/requirements.txt` cho ingestion)

### Phần A — Router + Multi-Agent
```bash
cd A2
python3 demo_multiturn.py      # demo hội thoại multi-turn qua 3 agent
```

### Phần B — Graph RAG + SGLang Generator
```bash
# Terminal 1: khởi động Generator (SGLang, AWQ)
python3 -m sglang.launch_server --model-path A1/generator_awq_v2 --port 30001 \
    --dtype float16 --quantization awq --attention-backend triton \
    --sampling-backend pytorch --mem-fraction-static 0.45

# Terminal 2: demo GraphRAG retrieval nối SSE streaming
cd A2
python3 demo_graphrag_stream.py
```

### Phần C — Cache thông minh + Guardrails
```bash
cd A2
python3 demo_part_c.py         # Demo end-to-end: cache pipeline (C2) nối tiếp guardrails (C3)
python3 test_guardrails.py     # C3: TTS preprocess, rate limit, circuit breaker, health check
python3 c2_benchmark.py        # C2.2: benchmark cache hit-rate + latency trên data thật
```

## Kết quả tóm tắt (xem chi tiết trong `docs/BaoCao_KyThuat_*.pdf`)

| Phần | Hạng mục nổi bật | Kết quả |
|---|---|---|
| A | Demo multi-turn qua 3 agent (order/consultant/faq) | PASS |
| A | Router fine-tune AWQ/fp16 | Checkpoint hỏng, dùng base model thay thế (xem báo cáo A) |
| B | GraphRAG hybrid search (dual-domain + expansion + rerank) | PASS |
| B | SSE streaming + TTFT thật | PASS (TTFT ~3.2s cold-start, cần đo lại warm) |
| B | Zero Hallucination | FAIL khi context rỗng — đã ghi nhận, cần thêm guardrail |
| C1 | Edge Deployment (GGUF/ARM) | Không thực hiện — thiếu thiết bị ARM |
| C2.1 | SLM Intent Extraction | PASS — fine-tune LoRA thật, accuracy 98.6% trên test set giữ riêng (≥90%) |
| C2.2 | Cache Pipeline & Paraphrase | PASS — hit-rate 90%, latency 23.8-46.3ms (yêu cầu ≤100ms) |
| C3 | Production Guardrails | PASS — 4/4 module có bằng chứng test thật |

Báo cáo được viết theo nguyên tắc **trung thực**: mọi PASS/FAIL đều dựa trên
log/số liệu đo được thật, không giả định hay làm tròn có lợi.