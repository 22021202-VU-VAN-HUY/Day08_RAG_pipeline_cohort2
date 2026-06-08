"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path
import re
import unicodedata

try:
    from src.task4_chunking_indexing import chunk_documents, load_documents
except ModuleNotFoundError:
    from task4_chunking_indexing import chunk_documents, load_documents

# TODO: Load corpus từ data/standardized/ hoặc từ vector store
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
BM25_INDEX = None


def _normalize_text(text: str) -> str:
    """Chuẩn hóa text để BM25 match ổn hơn với tiếng Việt có dấu/không dấu."""
    text = text.lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return "".join(char for char in text if unicodedata.category(char) != "Mn")


def tokenize(text: str) -> list[str]:
    """Tokenize đơn giản cho BM25, đủ ổn với dữ liệu markdown tiếng Việt."""
    normalized = _normalize_text(text)
    return re.findall(r"[a-z0-9]+", normalized)


def load_corpus() -> list[dict]:
    """Load corpus từ markdown chuẩn hóa và chunk giống Task 4."""
    global CORPUS
    if not CORPUS:
        documents = load_documents()
        CORPUS = chunk_documents(documents)
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def get_bm25_index():
    """Lazy-build BM25 index để import module nhanh và tái sử dụng giữa queries."""
    global BM25_INDEX
    corpus = load_corpus()
    if BM25_INDEX is None:
        BM25_INDEX = build_bm25_index(corpus)
    return BM25_INDEX


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if top_k <= 0:
        return []

    corpus = load_corpus()
    if not corpus:
        return []

    bm25 = get_bm25_index()
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue
        results.append({
            "content": corpus[index]["content"],
            "score": score,
            "metadata": dict(corpus[index].get("metadata", {})),
        })
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        preview = r["content"][:100].encode("ascii", "backslashreplace").decode("ascii")
        print(f"[{r['score']:.3f}] {preview}...")
