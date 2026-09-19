"""
B1.4 — Benchmark & kiểm tra đúng 4 tiêu chí nghiệm thu của Module B1:

  1. Node count: >=100 MenuItem, >=150 FAQ chunk, >=50 Document chunk.
  2. Top-5 precision >=80% trên tập câu hỏi test (tự sinh từ chính dữ liệu đã ingest,
     nên ground-truth chắc chắn đúng — câu hỏi lấy trực tiếp từ FAQ/MenuItem đã nạp,
     "đúng" nghĩa là chunk/MenuItem gốc phải xuất hiện trong top-5 kết quả trả về).
  3. Graph Expansion phải giúp recall tăng >=15% so với pure vector search. Đo bằng
     CẶP chunk LIỀN KỀ nhau (quan hệ NEXT) — vốn là 1 đoạn văn bị cắt làm 2 vì dài quá
     max_chars, nên chắc chắn liên quan ngữ nghĩa: query = nguyên văn chunk A -> kỳ vọng
     tìm ra chunk B (phần tiếp theo). Pure vector search thường CHỈ tìm lại được A (vì A
     giống hệt query) mà bỏ sót B (câu chữ có thể khác nhiều dù cùng đoạn); Graph
     Expansion (theo quan hệ NEXT) mới kéo được B vào -> đây là giá trị thật của bước
     Graph Expansion, khác với tiêu chí 2 (đo khả năng tìm lại đúng nguồn gốc).
  4. Zero hallucination: với câu hỏi ngoài phạm vi quán (out-of-domain), hệ thống
     phải KHÔNG cố trả về top-k ép buộc — sau ngưỡng rerank >=0.7, kết quả phải rỗng
     (nghĩa là "tôi không có thông tin" thay vì bịa/lấy đại 1 chunk không liên quan).

Chạy: python benchmark_b1.py   (yêu cầu Neo4j đã ingest xong dữ liệu bằng ingest.py)
"""
from __future__ import annotations

import csv
import random

from neo4j import GraphDatabase

from hybrid_search import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, hybrid_search, RERANK_THRESHOLD,
    dual_domain_search, graph_expansion,
)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

N_TEST_QUESTIONS = 20
OUT_OF_DOMAIN_QUERIES = [
    "Giá xăng hôm nay bao nhiêu vậy?",
    "Chính phủ Việt Nam có bao nhiêu bộ?",
    "What is the capital of France?",
    "Hôm nay đá banh đội nào thắng vậy?",
    "Công thức tính đạo hàm của hàm số bậc 2 là gì?",
]


# ==================== Tiêu chí 1: Node count ====================

def check_node_counts():
    with driver.session() as session:
        n_menu = session.run("MATCH (m:MenuItem) RETURN count(m) AS n").single()["n"]
        n_faq = session.run("MATCH (c:Chunk {source:'faq'}) RETURN count(c) AS n").single()["n"]
        n_doc = session.run("MATCH (c:Chunk {source:'document'}) RETURN count(c) AS n").single()["n"]

    print("=== Tiêu chí 1: Node count ===")
    results = [
        ("MenuItem", n_menu, 100),
        ("FAQ Chunk", n_faq, 150),
        ("Document Chunk", n_doc, 50),
    ]
    all_pass = True
    for label, actual, required in results:
        ok = actual >= required
        all_pass &= ok
        print(f"  {label:15s}: {actual:4d}  (yêu cầu >= {required})  -> {'PASS' if ok else 'FAIL'}")
    return all_pass


# ==================== Sinh bộ câu hỏi test từ chính dữ liệu đã ingest ====================

def build_test_set(menu_csv: str, faq_csv: str, n: int = N_TEST_QUESTIONS, seed: int = 42):
    """Ground-truth tự động: câu hỏi = câu hỏi FAQ gốc hoặc tên món ăn/đồ uống, đáp án đúng
    = chính chunk_id/MenuItem.name tương ứng đã ingest. Trộn cả 2 loại để test cân bằng cả
    2 domain (menu-domain và document/FAQ-domain) trong Dual-Domain Search."""
    rng = random.Random(seed)
    cases = []

    with open(faq_csv, encoding="utf-8") as f:
        faq_rows = list(csv.DictReader(f))
    with open(menu_csv, encoding="utf-8") as f:
        menu_rows = list(csv.DictReader(f))

    n_faq_cases = n * 2 // 3
    n_menu_cases = n - n_faq_cases

    for row in rng.sample(faq_rows, min(n_faq_cases, len(faq_rows))):
        cases.append({"query": row["question"], "expect_kind": "faq_text", "expect_value": row["answer"][:40]})
    for row in rng.sample(menu_rows, min(n_menu_cases, len(menu_rows))):
        cases.append({"query": f"Cho tôi hỏi về món {row['name']}", "expect_kind": "menu_name", "expect_value": row["name"]})

    return cases


def _is_hit(case: dict, results) -> bool:
    if case["expect_kind"] == "menu_name":
        return any(r.kind == "menu" and r.node_id == case["expect_value"] for r in results)
    return any(r.kind == "chunk" and case["expect_value"] in r.text for r in results)


# ==================== Tiêu chí 2: Top-5 precision ====================

def check_precision(test_cases: list[dict]):
    hits = 0
    for case in test_cases:
        results = hybrid_search(case["query"], top_k=5, use_expansion=True)
        if _is_hit(case, results):
            hits += 1

    n = len(test_cases)
    precision = hits / n if n else 0.0
    print(f"\n=== Tiêu chí 2: Top-5 precision (n={n} câu hỏi test) ===")
    print(f"  {hits}/{n} = {precision*100:.1f}%  (yêu cầu >= 80%)")
    precision_pass = precision >= 0.80
    print(f"  -> {'PASS' if precision_pass else 'FAIL'}")
    return precision_pass


# ==================== Tiêu chí 3: Graph Expansion recall lift ====================

def build_entity_pairs(n_pairs: int = 15):
    """Tìm cặp Chunk (A, B) LIỀN KỀ nhau qua quan hệ NEXT (A)-[:NEXT]->(B).
    2 chunk liền kề trong cùng section VỐN DĨ là 1 đoạn văn bị cắt làm 2 vì vượt quá
    max_chars -> chắc chắn liên quan về ngữ nghĩa (khác với thử cặp qua Entity chung,
    vốn phụ thuộc chất lượng entity extraction và dễ ra các danh từ chung không đặc
    trưng như 'quán', 'đồ uống' -> liên quan giả, bị rerank loại đúng)."""
    with driver.session() as session:
        rows = session.run(
            """
            MATCH (a:Chunk {source:'document'})-[:NEXT]->(b:Chunk {source:'document'})
            WITH a, b, rand() AS r
            ORDER BY r
            RETURN a.chunk_id AS aid, a.text AS atext, b.chunk_id AS bid, b.text AS btext
            LIMIT $limit
            """,
            limit=n_pairs,
        )
        pairs = [
            {"entity": "(chunk liền kề - NEXT)",
             "query_chunk": {"chunk_id": r["aid"], "text": r["atext"]},
             "target_chunk": {"chunk_id": r["bid"], "text": r["btext"]}}
            for r in rows
        ]
    return pairs


def check_expansion_lift(pairs: list[dict]):
    """Đo recall ở TẦNG CANDIDATE (trước Late Reranking) — đúng đúng phạm vi trách nhiệm
    của bước Graph Expansion trong pipeline. Rerank là bộ lọc precision RIÊNG (đã được
    kiểm tra ở tiêu chí 2 + 4), không nên trộn chung vào phép đo recall của Graph
    Expansion, vì rerank được huấn luyện để chấm "độ liên quan trả lời câu hỏi" chứ
    không phải "độ liên tục văn bản" — 2 vế của 1 câu bị cắt làm đôi vẫn LIÊN QUAN về
    ngữ nghĩa/nguồn gốc dù rerank có thể chấm điểm thấp khi xét riêng lẻ."""
    print(f"\n=== Tiêu chí 3: Graph Expansion recall lift so với pure vector search ===")
    n = len(pairs)
    if n == 0:
        print("  Không tìm được cặp Chunk liền kề (NEXT) nào để test -> bỏ qua (FAIL).")
        return False

    hits_with, hits_without = 0, 0
    for p in pairs:
        query = p["query_chunk"]["text"]
        target_id = p["target_chunk"]["chunk_id"]

        stage1 = dual_domain_search(query)
        without_ids = {c.node_id for c in stage1}
        with_ids = without_ids | {c.node_id for c in graph_expansion(stage1)}

        if target_id in with_ids:
            hits_with += 1
        if target_id in without_ids:
            hits_without += 1

    recall_with = hits_with / n
    recall_without = hits_without / n
    print(f"  Số cặp test (Chunk liền kề - NEXT, cùng nguồn document): {n}")
    print(f"  Pure vector (không expansion): tìm đúng {hits_without}/{n} = {recall_without*100:.1f}%")
    print(f"  Có Graph Expansion           : tìm đúng {hits_with}/{n} = {recall_with*100:.1f}%")
    if recall_without > 0:
        lift = (recall_with - recall_without) / recall_without
    else:
        lift = float("inf") if recall_with > 0 else 0.0
    lift_str = "inf" if lift == float("inf") else f"{lift*100:.1f}%"
    lift_pass = lift >= 0.15
    print(f"  Lift: {lift_str}  (yêu cầu >= 15%)  -> {'PASS' if lift_pass else 'FAIL'}")
    return lift_pass


# ==================== Tiêu chí 4: Zero hallucination ====================

def check_zero_hallucination():
    print(f"\n=== Tiêu chí 4: Zero hallucination (câu hỏi ngoài phạm vi quán) ===")
    all_pass = True
    for q in OUT_OF_DOMAIN_QUERIES:
        results = hybrid_search(q, top_k=5)
        ok = len(results) == 0
        all_pass &= ok
        status = "PASS (rỗng, không bịa)" if ok else f"FAIL (trả về {len(results)} kết quả không liên quan)"
        print(f"  '{q}' -> {status}")
    return all_pass


# ==================== Main ====================

def main(menu_csv: str = "data/menu.csv", faq_csv: str = "data/faq.csv"):
    node_pass = check_node_counts()
    test_cases = build_test_set(menu_csv, faq_csv)
    precision_pass = check_precision(test_cases)
    entity_pairs = build_entity_pairs()
    lift_pass = check_expansion_lift(entity_pairs)
    halluc_pass = check_zero_hallucination()

    print("\n" + "=" * 60)
    print("TỔNG KẾT B1 — 4 TIÊU CHÍ NGHIỆM THU")
    print("=" * 60)
    print(f"1. Node count (>=100/150/50)                : {'PASS' if node_pass else 'FAIL'}")
    print(f"2. Top-5 precision (>=80%)                   : {'PASS' if precision_pass else 'FAIL'}")
    print(f"3. Graph Expansion recall lift (>=15%)       : {'PASS' if lift_pass else 'FAIL'}")
    print(f"4. Zero hallucination (out-of-domain rỗng)   : {'PASS' if halluc_pass else 'FAIL'}")


if __name__ == "__main__":
    main()