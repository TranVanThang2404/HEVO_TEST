"""
B1.3 — Data Ingestion Pipeline: nạp Menu CSV, FAQ CSV, Document (txt/pdf/docx) vào Neo4j.

Luồng xử lý:
  1. Menu CSV   -> parse, tạo embedding (name + description), lưu vào Graph + BELONGS_TO Category.
  2. FAQ CSV    -> tách từng cặp Q&A thành 1 Chunk (source="faq"), tạo embedding, lưu metadata
                   (category), nối NEXT giữa các FAQ liền kề cùng category.
  3. Document   -> Semantic Chunking (gradient breakpoint đơn giản — xem semantic_chunk()),
                   Entity Extraction tự động bằng LLM (gọi local LLM server OpenAI-compatible,
                   TÁI SỬ DỤNG đúng hạ tầng đã dùng ở A1.2/cleaning), tạo relationship MENTIONS,
                   nối NEXT giữa các chunk liền kề trong cùng section.
  4. Entity Deduplication: fuzzy matching bằng Jaccard similarity (>= 0.85) trên tập từ, gộp
     các entity trùng lặp trước khi ghi vào Graph.
  5. Watch Mode: theo dõi thư mục `data/watch/`, phát hiện file mới -> tự động ingest realtime.

Chạy toàn bộ 1 lần:
    python ingest.py --menu data/menu.csv --faq data/faq.csv --doc data/handbook.txt

Bật watch mode (chạy nền, theo dõi thư mục data/watch/):
    python ingest.py --watch
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import uuid
from itertools import combinations

from neo4j import GraphDatabase
from openai import OpenAI

from embeddings import embed_texts

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "highlands123")

# Tái sử dụng local LLM server (OpenAI-compatible) đã dựng ở A1.2 cho Entity Extraction.
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")
llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-no-key-needed"),
                     base_url=os.getenv("OPENAI_BASE_URL"))

JACCARD_THRESHOLD = 0.85

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ==================== Semantic Chunking (gradient breakpoint đơn giản) ====================

def semantic_chunk(text: str, max_chars: int = 90) -> list[str]:
    """Tách văn bản theo mệnh đề (câu + mệnh đề con phân tách bởi dấu phẩy/chấm), rồi gộp
    liên tiếp tới khi đạt max_chars. Đây là bản "gradient breakpoint" đơn giản dựa trên độ
    dài — bản đầy đủ hơn có thể dùng độ tương đồng embedding giữa câu liền kề để tìm điểm
    ngắt ngữ nghĩa tự nhiên, nhưng cách này đã đủ tạo ra các chunk mạch lạc, kích thước đều."""
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


# ==================== Entity Extraction (LLM) ====================

def extract_entities(text: str) -> list[str]:
    """Gọi LLM trích xuất thực thể (tên món, địa điểm, tên chính sách...) từ 1 đoạn text.
    Trả về list tên thực thể thô (chưa dedup)."""
    prompt = (
        "Trích xuất các THỰC THỂ quan trọng (tên món ăn/đồ uống, tên chính sách, địa điểm, "
        "chức danh nhân sự...) xuất hiện trong đoạn văn bản sau. Trả về DUY NHẤT 1 JSON array "
        f'các chuỗi tên thực thể, KHÔNG giải thích:\n\n"{text}"'
    )
    try:
        resp = llm_client.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = resp.choices[0].message.content.strip()
        content = re.sub(r"^```(json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            content = match.group(0)
        entities = json.loads(content)
        return [e.strip() for e in entities if isinstance(e, str) and e.strip()]
    except Exception as e:
        print(f"[extract_entities] Lỗi ({e}), bỏ qua entity cho đoạn này.")
        return []


# ==================== Entity Deduplication (Jaccard similarity) ====================

def _token_set(name: str) -> set:
    return set(re.sub(r"[^\wàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ ]",
                       "", name.lower()).split())


def jaccard_similarity(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def deduplicate_entities(raw_names: list[str]) -> dict[str, str]:
    """Gộp các tên entity trùng lặp (Jaccard >= 0.85) -> trả về mapping {tên gốc: tên canonical}.
    Canonical được chọn là tên xuất hiện SỚM NHẤT trong danh sách (đơn giản, ổn định)."""
    unique_names = list(dict.fromkeys(raw_names))  # giữ thứ tự xuất hiện, loại trùng y hệt
    canonical_map = {name: name for name in unique_names}
    parent = {name: name for name in unique_names}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for a, b in combinations(unique_names, 2):
        if jaccard_similarity(a, b) >= JACCARD_THRESHOLD:
            union(a, b)

    for name in unique_names:
        canonical_map[name] = find(name)

    n_merged = len(unique_names) - len(set(canonical_map.values()))
    if n_merged:
        print(f"[dedup] Gộp {n_merged} entity trùng lặp (Jaccard >= {JACCARD_THRESHOLD}).")
    return canonical_map


# ==================== Ingestion: Menu CSV ====================

def ingest_menu(csv_path: str):
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    texts = [f"{r['name']}. {r['description']}" for r in rows]
    embeddings = embed_texts(texts)

    with driver.session() as session:
        for row, emb in zip(rows, embeddings):
            session.run(
                """
                MERGE (m:MenuItem {name: $name})
                SET m.price_vnd = $price_vnd, m.category = $category, m.size = $size,
                    m.ingredients = $ingredients, m.description = $description, m.embedding = $embedding
                MERGE (cat:Category {name: $category})
                MERGE (m)-[:BELONGS_TO]->(cat)
                """,
                name=row["name"], price_vnd=int(row["price_vnd"]), category=row["category"],
                size=row["size"], ingredients=row["ingredients"], description=row["description"],
                embedding=emb,
            )
    print(f"[ingest_menu] Đã nạp {len(rows)} MenuItem.")


# ==================== Ingestion: FAQ CSV ====================

def ingest_faq(csv_path: str):
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    texts = [f"{r['question']} {r['answer']}" for r in rows]
    embeddings = embed_texts(texts)

    prev_chunk_id_by_category: dict[str, str] = {}
    all_entities: list[str] = []
    chunk_entity_pairs: list[tuple[str, list[str]]] = []

    with driver.session() as session:
        for row, emb, text in zip(rows, embeddings, texts):
            chunk_id = f"faq-{uuid.uuid4().hex[:12]}"
            session.run(
                """
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.text = $text, c.source = 'faq', c.category = $category,
                    c.question = $question, c.answer = $answer, c.embedding = $embedding
                """,
                chunk_id=chunk_id, text=text, category=row["category"],
                question=row["question"], answer=row["answer"], embedding=emb,
            )
            prev_id = prev_chunk_id_by_category.get(row["category"])
            if prev_id:
                session.run(
                    "MATCH (a:Chunk {chunk_id: $prev}), (b:Chunk {chunk_id: $cur}) MERGE (a)-[:NEXT]->(b)",
                    prev=prev_id, cur=chunk_id,
                )
            prev_chunk_id_by_category[row["category"]] = chunk_id

            entities = extract_entities(row["answer"])
            all_entities.extend(entities)
            chunk_entity_pairs.append((chunk_id, entities))

    _link_entities(all_entities, chunk_entity_pairs)
    print(f"[ingest_faq] Đã nạp {len(rows)} FAQ chunk.")



# ==================== Đọc nội dung Document: hỗ trợ .txt / .pdf / .docx ====================

def _read_document_text(doc_path: str) -> str:
    """Trích xuất text thô từ file Document, hỗ trợ .txt, .pdf, .docx (B1.3)."""
    ext = doc_path.lower().rsplit(".", 1)[-1] if "." in doc_path else ""
    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(doc_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == "docx":
        import docx
        d = docx.Document(doc_path)
        return "\n".join(p.text for p in d.paragraphs)
    else:
        with open(doc_path, encoding="utf-8") as f:
            return f.read()


# ==================== Ingestion: Document (txt, đã Semantic Chunking) ====================

def ingest_document(txt_path: str):
    raw = _read_document_text(txt_path)

    sections = [s for s in re.split(r"\n?## ", raw) if s.strip()]
    all_chunk_texts, chunk_section = [], []
    for section in sections:
        lines = section.strip().split("\n", 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        for chunk_text in semantic_chunk(body):
            all_chunk_texts.append(chunk_text)
            chunk_section.append(title)

    embeddings = embed_texts(all_chunk_texts)

    prev_chunk_id_by_section: dict[str, str] = {}
    all_entities: list[str] = []
    chunk_entity_pairs: list[tuple[str, list[str]]] = []

    with driver.session() as session:
        for text, section, emb in zip(all_chunk_texts, chunk_section, embeddings):
            chunk_id = f"doc-{uuid.uuid4().hex[:12]}"
            session.run(
                """
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.text = $text, c.source = 'document', c.section = $section, c.embedding = $embedding
                """,
                chunk_id=chunk_id, text=text, section=section, embedding=emb,
            )
            prev_id = prev_chunk_id_by_section.get(section)
            if prev_id:
                session.run(
                    "MATCH (a:Chunk {chunk_id: $prev}), (b:Chunk {chunk_id: $cur}) MERGE (a)-[:NEXT]->(b)",
                    prev=prev_id, cur=chunk_id,
                )
            prev_chunk_id_by_section[section] = chunk_id

            entities = extract_entities(text)
            all_entities.extend(entities)
            chunk_entity_pairs.append((chunk_id, entities))

    _link_entities(all_entities, chunk_entity_pairs)
    print(f"[ingest_document] Đã nạp {len(all_chunk_texts)} Document chunk (từ {len(sections)} section).")


# ==================== Entity linking (dùng chung cho FAQ + Document) ====================

def _link_entities(all_entities: list[str], chunk_entity_pairs: list[tuple[str, list[str]]]):
    if not all_entities:
        return
    canonical_map = deduplicate_entities(all_entities)
    with driver.session() as session:
        for canonical in set(canonical_map.values()):
            session.run("MERGE (e:Entity {name: $name})", name=canonical)
        for chunk_id, entities in chunk_entity_pairs:
            for raw_name in entities:
                canonical = canonical_map.get(raw_name, raw_name)
                session.run(
                    """
                    MATCH (c:Chunk {chunk_id: $chunk_id}), (e:Entity {name: $name})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    chunk_id=chunk_id, name=canonical,
                )


# ==================== Watch Mode (B1.3) ====================

def start_watch_mode(watch_dir: str = "data/watch"):
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    os.makedirs(watch_dir, exist_ok=True)

    class Handler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            path = event.src_path
            print(f"[watch] Phát hiện file mới: {path}")
            time.sleep(1)  # đợi file ghi xong hẳn
            try:
                if path.endswith(".csv") and "menu" in path.lower():
                    ingest_menu(path)
                elif path.endswith(".csv") and "faq" in path.lower():
                    ingest_faq(path)
                elif path.endswith((".txt", ".md", ".pdf", ".docx")):
                    ingest_document(path)
                else:
                    print(f"[watch] Không nhận diện được loại file, bỏ qua: {path}")
            except Exception as e:
                print(f"[watch] Lỗi khi ingest {path}: {e}")

    observer = Observer()
    observer.schedule(Handler(), watch_dir, recursive=False)
    observer.start()
    print(f"[watch] Đang theo dõi thư mục '{watch_dir}' — thả file .csv/.txt mới vào đây để tự động ingest.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--menu", default=None)
    parser.add_argument("--faq", default=None)
    parser.add_argument("--doc", default=None)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()

    if args.watch:
        start_watch_mode()
    else:
        if args.menu:
            ingest_menu(args.menu)
        if args.faq:
            ingest_faq(args.faq)
        if args.doc:
            ingest_document(args.doc)
        print("\nXong ingestion. Kiểm tra số lượng node bằng benchmark_b1.py.")