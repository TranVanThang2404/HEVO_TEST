"""Debug nhanh: in ra các cặp Entity-chung tìm được, và điểm rerank thực tế của target
chunk so với ngưỡng 0.7, để biết bị loại vì rerank threshold hay vì graph_expansion
không tìm ra candidate đó ngay từ đầu."""
from benchmark_b1 import build_entity_pairs
from hybrid_search import dual_domain_search, graph_expansion, RERANK_THRESHOLD
from embeddings import rerank

pairs = build_entity_pairs()
print(f"Tổng số cặp tìm được: {len(pairs)}\n")

for p in pairs:
    query = p["query_chunk"]["text"]
    target_id = p["target_chunk"]["chunk_id"]
    print(f"--- Entity: {p['entity']} ---")
    print(f"Query (chunk A): {query[:100]}")
    print(f"Target (chunk B, id={target_id}): {p['target_chunk']['text'][:100]}")

    stage1 = dual_domain_search(query)
    expanded = graph_expansion(stage1)
    all_candidates = stage1 + expanded

    target_in_candidates = next((c for c in all_candidates if c.node_id == target_id), None)
    if target_in_candidates is None:
        print("=> Target KHÔNG có trong candidate list (kể cả sau graph_expansion) -> lỗi ở graph_expansion, không phải rerank.")
    else:
        score = rerank(query, [target_in_candidates.text])[0]
        print(f"=> Target CÓ trong candidate list. Điểm rerank = {score:.3f} (ngưỡng {RERANK_THRESHOLD}) "
              f"-> {'qua ngưỡng' if score >= RERANK_THRESHOLD else 'BỊ LOẠI bởi ngưỡng rerank'}")
    print()