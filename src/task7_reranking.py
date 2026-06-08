"""
Task 7 — Reranking Module.

Phương pháp chọn cho project này: MMR (Maximal Marginal Relevance).

Lý do chọn:
    - Dữ liệu pháp luật có nhiều đoạn dài và lặp cấu trúc, nên top results từ
      semantic/BM25 dễ bị trùng nội dung.
    - MMR vừa giữ độ liên quan với query, vừa phạt những candidate quá giống
      các kết quả đã chọn, giúp context cuối đa dạng hơn.
    - MMR chạy offline, không cần API key/cross-encoder nặng, phù hợp môi
      trường hiện tại của project.

Cơ chế:
    MMR(doc) = λ * relevance(query, doc)
               - (1 - λ) * max_similarity(doc, selected_docs)

    λ gần 1.0 ưu tiên relevance; λ thấp hơn tăng diversity.
    Ở đây dùng λ=0.7 để vẫn ưu tiên đúng query nhưng giảm trùng lặp rõ rệt.
"""

import re
import unicodedata

try:
    from src.task4_chunking_indexing import cosine_similarity, embed_text
except ModuleNotFoundError:
    from task4_chunking_indexing import cosine_similarity, embed_text


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if top_k <= 0 or not candidates:
        return []

    prepared_candidates = _ensure_candidate_embeddings(candidates)
    selected_indices: list[int] = []
    remaining_indices = list(range(len(prepared_candidates)))

    while remaining_indices and len(selected_indices) < top_k:
        best_index = remaining_indices[0]
        best_mmr_score = float("-inf")
        best_relevance = 0.0

        for index in remaining_indices:
            candidate_embedding = prepared_candidates[index]["embedding"]
            relevance = _candidate_similarity(query_embedding, candidate_embedding)

            if selected_indices:
                max_similarity_to_selected = max(
                    _candidate_similarity(candidate_embedding, prepared_candidates[selected]["embedding"])
                    for selected in selected_indices
                )
            else:
                max_similarity_to_selected = 0.0

            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * max_similarity_to_selected
            )

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_index = index
                best_relevance = relevance

        selected_indices.append(best_index)
        remaining_indices.remove(best_index)
        prepared_candidates[best_index]["score"] = float(best_mmr_score)
        prepared_candidates[best_index]["rerank_score"] = float(best_mmr_score)
        prepared_candidates[best_index]["relevance_score"] = float(best_relevance)

    return [
        _strip_embedding(prepared_candidates[index])
        for index in selected_indices
    ]


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k_value: float = 60.0,
) -> list[dict]:
    """
    Reciprocal Rank Fusion.

    Args:
        ranked_lists: Danh sách các ranking output từ semantic/lexical search.
        top_k: Số lượng kết quả tối đa.
        k_value: Hằng số khử nghịch đảo rank.

    Returns:
        List of merged candidates with combined scores.
    """
    if top_k <= 0 or not ranked_lists:
        return []

    aggregated: dict[str, dict] = {}

    def _make_key(item: dict) -> str:
        metadata = item.get("metadata") or {}
        path = str(metadata.get("path", ""))
        return f"{item.get('content', '').strip()}||{path}"

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = _make_key(item)
            if key not in aggregated:
                aggregated[key] = {
                    "content": item.get("content", ""),
                    "metadata": dict(item.get("metadata", {})),
                    "rrf_score": 0.0,
                    "original_scores": [],
                }

            aggregated[key]["rrf_score"] += 1.0 / (k_value + rank)
            if item.get("score") is not None:
                aggregated[key]["original_scores"].append(float(item["score"]))

    merged = []
    for item in aggregated.values():
        original_scores = item["original_scores"]
        combined_score = max(original_scores) if original_scores else item["rrf_score"]
        merged.append({
            "content": item["content"],
            "metadata": item["metadata"],
            "score": float(combined_score),
            "rrf_score": float(item["rrf_score"]),
        })

    merged.sort(
        key=lambda entry: (entry["rrf_score"], entry["score"]),
        reverse=True,
    )
    return merged[:top_k]


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "mmr",
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method != "mmr":
        raise ValueError("Task 7 hiện chọn MMR; hãy gọi rerank(..., method='mmr').")

    query_embedding = _embed_text_with_fallback(query)
    return rerank_mmr(query_embedding, candidates, top_k=top_k)


def _normalize_text(text: str) -> str:
    text = text.lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return "".join(char for char in text if unicodedata.category(char) != "Mn")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize_text(text))


def _lexical_embedding(text: str) -> dict[str, float]:
    """Fallback sparse embedding nếu model sentence-transformers chưa sẵn sàng."""
    tokens = _tokenize(text)
    if not tokens:
        return {}

    counts: dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0

    norm = sum(value * value for value in counts.values()) ** 0.5
    if norm == 0:
        return counts
    return {token: value / norm for token, value in counts.items()}


def _sparse_cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def _embed_text_with_fallback(text: str):
    try:
        return embed_text(text)
    except Exception:
        return _lexical_embedding(text)


def _candidate_similarity(left_embedding, right_embedding) -> float:
    if isinstance(left_embedding, dict) and isinstance(right_embedding, dict):
        return _sparse_cosine(left_embedding, right_embedding)
    if isinstance(left_embedding, list) and isinstance(right_embedding, list):
        return cosine_similarity(left_embedding, right_embedding)
    return 0.0


def _ensure_candidate_embeddings(candidates: list[dict]) -> list[dict]:
    prepared_candidates = []
    for candidate in candidates:
        item = dict(candidate)
        item["metadata"] = dict(candidate.get("metadata", {}))
        if "embedding" not in item or item["embedding"] is None:
            item["embedding"] = _embed_text_with_fallback(item.get("content", ""))
        prepared_candidates.append(item)
    return prepared_candidates


def _strip_embedding(candidate: dict) -> dict:
    item = dict(candidate)
    item.pop("embedding", None)
    return item


def _safe_display(text: str) -> str:
    return text.encode("ascii", "backslashreplace").decode("ascii")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {_safe_display(r['content'])}")
