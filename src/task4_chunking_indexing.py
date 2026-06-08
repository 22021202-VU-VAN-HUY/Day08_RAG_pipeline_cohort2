"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path
import json
import re

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
INDEX_PATH = INDEX_DIR / "local_vector_store.json"
_EMBEDDING_MODEL_INSTANCE = None


# =============================================================================
# CẤU HÌNH — Giải thích lựa chọn theo yêu cầu Task 4
# =============================================================================

# Dùng recursive chunking vì dữ liệu gồm cả văn bản pháp luật và bài báo; các
# file DOCX sau khi convert sang markdown không phải lúc nào cũng giữ heading
# sạch, nên tách theo heading có thể bỏ sót cấu trúc. CHUNK_SIZE=500 giúp mỗi
# chunk đủ ngắn để retrieval/citation tập trung; CHUNK_OVERLAP=75 giữ ngữ cảnh
# ở ranh giới giữa hai chunk để tránh mất ý khi câu bị cắt.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
CHUNKING_METHOD = "recursive"

# Embedding model dùng sentence-transformers/all-MiniLM-L6-v2 với
# EMBEDDING_DIM=384. Đây là model nhẹ, tốc độ nhanh, dễ chạy local cho demo và
# vẫn tạo dense vector thật để semantic search bằng cosine similarity. Với tiếng
# Việt chuyên sâu có thể đổi sang BAAI/bge-m3, nhưng MiniLM phù hợp hơn khi cần
# cài đặt nhanh và tài nguyên máy hạn chế.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Vector store dùng local_json: index_to_vectorstore() lưu toàn bộ chunks của
# toàn bộ documents đã load vào data/index/local_vector_store.json để test tái
# lập được. Weaviate vẫn là lựa chọn phù hợp cho production/demo khi có Docker
# hoặc Weaviate Cloud.
VECTOR_STORE = "local_json"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue

        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": str(relative_path).replace("\\", "/"),
                "type": doc_type,
            },
        })

    return documents


def _find_split_boundary(text: str, start: int, end: int) -> int:
    """Find a natural split point without exceeding CHUNK_SIZE."""
    if end >= len(text):
        return len(text)

    min_boundary = start + int(CHUNK_SIZE * 0.5)
    best_boundary = -1
    best_separator_len = 0
    for separator in ("\n\n", "\n", ". ", "; ", ", ", " "):
        boundary = text.rfind(separator, start, end)
        if boundary > min_boundary and boundary > best_boundary:
            best_boundary = boundary
            best_separator_len = len(separator)

    if best_boundary != -1:
        return best_boundary + best_separator_len
    return end


def _split_text_recursive(text: str) -> list[str]:
    """Split text into overlapping chunks with stable max size."""
    normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
    chunks = []
    start = 0
    previous_start = -1

    while start < len(normalized) and start != previous_start:
        previous_start = start
        hard_end = min(start + CHUNK_SIZE, len(normalized))
        end = _find_split_boundary(normalized, start, hard_end)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(normalized):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)
        while start < len(normalized) and normalized[start].isspace():
            start += 1

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunks = []
    for doc in documents:
        splits = _split_text_recursive(doc["content"])
        for index, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **doc.get("metadata", {}),
                    "chunk_index": index,
                    "chunking_method": CHUNKING_METHOD,
                },
            })
    return chunks


def get_embedding_model():
    """Load SentenceTransformer model once and reuse it."""
    global _EMBEDDING_MODEL_INSTANCE
    if _EMBEDDING_MODEL_INSTANCE is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDING_MODEL_INSTANCE = SentenceTransformer(
            EMBEDDING_MODEL,
            local_files_only=True,
        )
    return _EMBEDDING_MODEL_INSTANCE


def embed_text(text: str) -> list[float]:
    """Create a MiniLM embedding normalized for cosine similarity."""
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    model = get_embedding_model()
    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    embedded = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunk = {
            "content": chunk["content"],
            "metadata": dict(chunk.get("metadata", {})),
            "embedding": embedding.tolist(),
        }
        embedded.append(embedded_chunk)
    return embedded


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunking_method": CHUNKING_METHOD,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "vector_store": VECTOR_STORE,
        },
        "chunks": chunks,
    }
    INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return INDEX_PATH


def run_pipeline():
    """Chạy toàn bộ pipeline: load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_path = index_to_vectorstore(chunks)
    print(f"Indexed to vector store: {index_path}")


if __name__ == "__main__":
    run_pipeline()
