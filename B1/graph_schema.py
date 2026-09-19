"""
B1.1 — Graph Database Schema (Neo4j).

Node types (tối thiểu theo đề bài):
  - MenuItem  : name, price_vnd, category, size, ingredients, description, embedding
  - Chunk     : text, source ("faq" | "document"), section, embedding
  - Entity    : name, type (tên món / địa điểm / chính sách...)

Relationships:
  - (:Chunk)-[:NEXT]->(:Chunk)         giữa các Chunk liền kề (theo thứ tự trong tài liệu gốc)
  - (:Chunk)-[:MENTIONS]->(:Entity)    Chunk nhắc tới Entity nào
  - (:MenuItem)-[:BELONGS_TO]->(:Category)

Vector Index (B1.2):
  - menu_embedding   trên MenuItem.embedding
  - chunk_embedding  trên Chunk.embedding

Chạy: python graph_schema.py   (tạo constraints + vector index; idempotent, chạy lại
không lỗi nếu đã tồn tại).
"""
import os
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "highlands123")

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))  # khớp với chiều vector của embedding model dùng ở embeddings.py

SCHEMA_STATEMENTS = [
    # Constraints (đảm bảo duy nhất, đồng thời tự tạo index thường cho các field này)
    "CREATE CONSTRAINT menuitem_name IF NOT EXISTS FOR (m:MenuItem) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (cat:Category) REQUIRE cat.name IS UNIQUE",
]

VECTOR_INDEX_STATEMENTS = [
    f"""
    CREATE VECTOR INDEX menu_embedding IF NOT EXISTS
    FOR (m:MenuItem) ON (m.embedding)
    OPTIONS {{ indexConfig: {{
        `vector.dimensions`: {EMBEDDING_DIM},
        `vector.similarity_function`: 'cosine'
    }} }}
    """,
    f"""
    CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
    FOR (c:Chunk) ON (c.embedding)
    OPTIONS {{ indexConfig: {{
        `vector.dimensions`: {EMBEDDING_DIM},
        `vector.similarity_function`: 'cosine'
    }} }}
    """,
]


def setup_schema():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        for stmt in SCHEMA_STATEMENTS:
            session.run(stmt)
            print(f"[schema] OK: {stmt.strip().splitlines()[0]}")
        for stmt in VECTOR_INDEX_STATEMENTS:
            session.run(stmt)
            print(f"[schema] OK: vector index tạo/đã tồn tại")
    driver.close()
    print("\nSchema + Vector Index đã sẵn sàng trên Neo4j.")


if __name__ == "__main__":
    setup_schema()