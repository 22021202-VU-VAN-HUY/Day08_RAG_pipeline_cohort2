"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


import json

try:
    from src.task4_chunking_indexing import (
        EMBEDDING_MODEL,
        INDEX_PATH,
        chunk_documents,
        cosine_similarity,
        embed_chunks,
        embed_text,
        index_to_vectorstore,
        load_documents,
    )
except ModuleNotFoundError:
    from task4_chunking_indexing import (
        EMBEDDING_MODEL,
        INDEX_PATH,
        chunk_documents,
        cosine_similarity,
        embed_chunks,
        embed_text,
        index_to_vectorstore,
        load_documents,
    )


def _load_or_build_index() -> list[dict]:
    """Load local vector index, building it from standardized markdown if needed."""
    if INDEX_PATH.exists():
        payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        chunks = payload.get("chunks", [])
        if chunks and config.get("embedding_model") == EMBEDDING_MODEL:
            return chunks

    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    return embedded_chunks


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if top_k <= 0:
        return []

    query_embedding = embed_text(query)
    chunks = _load_or_build_index()
    results = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk.get("embedding", []))
        results.append({
            "content": chunk.get("content", ""),
            "score": float(score),
            "metadata": dict(chunk.get("metadata", {})),
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _safe_display(text: str) -> str:
    return text.encode("ascii", "backslashreplace").decode("ascii")


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        preview = _safe_display(r["content"][:100])
        print(f"[{r['score']:.3f}] {preview}...")
