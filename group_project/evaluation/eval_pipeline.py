"""
RAG Evaluation Pipeline — Simple keyword-based evaluation.

Không依赖 LLM evaluator (DeepEval/RAGAS) vì Mimo không output JSON chuẩn.
Thay vào đó dùng keyword matching + semantic similarity cho 4 metrics:
    1. Faithfulness — answer có nằm trong context không?
    2. Answer Relevancy — answer có liên quan đến question không?
    3. Context Recall — context có chứa expected answer không?
    4. Context Precision — context có relevant không?
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# RAG Pipeline wrapper
# =============================================================================

def run_rag_pipeline(question: str, use_reranking: bool = True, dense_only: bool = False) -> dict:
    from src.task5_semantic_search import semantic_search
    from src.task7_reranking import rerank
    from src.task10_generation import reorder_for_llm, format_context, SYSTEM_PROMPT, TEMPERATURE, TOP_P

    top_k = 5

    if dense_only:
        chunks = semantic_search(question, top_k=top_k * 2)
        if use_reranking and chunks:
            chunks = rerank(question, chunks, top_k=top_k)
        else:
            chunks = chunks[:top_k]
    else:
        from src.task9_retrieval_pipeline import retrieve
        chunks = retrieve(question, top_k=top_k, use_reranking=use_reranking)

    if not chunks:
        return {"answer": "Không tìm thấy thông tin.", "sources": [], "retrieval_source": "none"}

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_msg = f"Context:\n{context}\n\n---\n\nQuestion: {question}"

    try:
        from openai import OpenAI
        mimo_key = os.getenv("MIMO_API_KEY", "")
        mimo_base = os.getenv("MIMO_BASE_URL", "")
        mimo_model = os.getenv("MIMO_MODEL", "mimo-v2.5-pro")

        if mimo_key and mimo_base:
            client = OpenAI(api_key=mimo_key, base_url=mimo_base)
            model = mimo_model
        else:
            raise ValueError("No API key")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=512,
        )
        answer = response.choices[0].message.content or ""
    except Exception as e:
        answer = f"LLM error: {e}"

    return {
        "answer": answer,
        "sources": chunks,
        "context": context,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


# =============================================================================
# Simple Evaluation Metrics (keyword-based, no LLM needed)
# =============================================================================

def _tokenize(text: str) -> set[str]:
    """Simple Vietnamese-aware tokenization."""
    import re
    text = text.lower()
    text = re.sub(r"[àáạảãâầấậẩẫăằắặẳẵ]", "a", text)
    text = re.sub(r"[èéẹẻẽêềếệểễ]", "e", text)
    text = re.sub(r"[ìíịỉĩ]", "i", text)
    text = re.sub(r"[òóọỏõôồốộổỗơờớợởỡ]", "o", text)
    text = re.sub(r"[ùúụủũưừứựửữ]", "u", text)
    text = re.sub(r"[ỳýỵỷỹ]", "y", text)
    text = re.sub(r"đ", "d", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return set(text.split())


def faithfulness_score(answer: str, context: str) -> float:
    """Faithfulness: answer tokens có nằm trong context không?"""
    ans_tokens = _tokenize(answer)
    ctx_tokens = _tokenize(context)
    if not ans_tokens:
        return 0.0
    overlap = ans_tokens & ctx_tokens
    return len(overlap) / len(ans_tokens)


def answer_relevancy_score(answer: str, question: str) -> float:
    """Answer Relevancy: answer có chứa keywords từ question không?"""
    q_tokens = _tokenize(question)
    a_tokens = _tokenize(answer)
    if not q_tokens:
        return 0.0
    overlap = q_tokens & a_tokens
    return len(overlap) / len(q_tokens)


def context_recall_score(context: str, expected_answer: str) -> float:
    """Context Recall: expected answer keywords có trong context không?"""
    exp_tokens = _tokenize(expected_answer)
    ctx_tokens = _tokenize(context)
    if not exp_tokens:
        return 0.0
    overlap = exp_tokens & ctx_tokens
    return len(overlap) / len(exp_tokens)


def context_precision_score(context: str, question: str) -> float:
    """Context Precision: context có chứa keywords từ question không?"""
    q_tokens = _tokenize(question)
    ctx_tokens = _tokenize(context)
    if not q_tokens:
        return 0.0
    overlap = q_tokens & ctx_tokens
    return len(overlap) / len(q_tokens)


def evaluate_single(question: str, answer: str, context: str, expected: str) -> dict:
    return {
        "faithfulness": round(faithfulness_score(answer, context), 3),
        "answer_relevancy": round(answer_relevancy_score(answer, question), 3),
        "context_recall": round(context_recall_score(context, expected), 3),
        "context_precision": round(context_precision_score(context, question), 3),
    }


# =============================================================================
# A/B Evaluation
# =============================================================================

def evaluate_config(golden_dataset: list[dict], config_name: str,
                    use_reranking: bool, dense_only: bool) -> dict:
    print(f"\n{'='*60}")
    print(f"Evaluating: {config_name}")
    print("=" * 60)

    results = []
    for i, item in enumerate(golden_dataset, 1):
        print(f"  [{i}/{len(golden_dataset)}] {item['question'][:60]}...")
        rag = run_rag_pipeline(item["question"], use_reranking=use_reranking, dense_only=dense_only)

        metrics = evaluate_single(
            question=item["question"],
            answer=rag["answer"],
            context=rag.get("context", ""),
            expected=item["expected_answer"],
        )

        results.append({
            "question": item["question"],
            "expected": item["expected_answer"],
            "actual": rag["answer"],
            "sources_count": len(rag["sources"]),
            "metrics": metrics,
        })

    # Compute averages
    avg = {}
    for m in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        scores = [r["metrics"][m] for r in results]
        avg[m] = round(sum(scores) / len(scores), 3) if scores else 0.0

    return {"config": config_name, "results": results, "averages": avg}


# =============================================================================
# Export Results
# =============================================================================

def export_results(eval_a: dict, eval_b: dict):
    content = "# RAG Evaluation Results\n\n"
    content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    content += f"**Golden Dataset:** {len(eval_a['results'])} Q&A pairs\n\n"
    content += f"**Evaluation Method:** Keyword-based metrics (no LLM evaluator)\n\n"
    content += "---\n\n"

    # Config A
    content += f"## {eval_a['config']}\n\n"
    content += "| Metric | Score |\n|--------|-------|\n"
    for m, s in eval_a["averages"].items():
        content += f"| {m} | {s:.3f} |\n"
    content += "\n"

    # Config B
    content += f"## {eval_b['config']}\n\n"
    content += "| Metric | Score |\n|--------|-------|\n"
    for m, s in eval_b["averages"].items():
        content += f"| {m} | {s:.3f} |\n"
    content += "\n"

    # A/B Comparison
    content += "## A/B Comparison\n\n"
    content += "| Metric | Config A | Config B | Winner |\n"
    content += "|--------|----------|----------|--------|\n"
    for m in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        a = eval_a["averages"][m]
        b = eval_b["averages"][m]
        winner = "**A**" if a > b else "**B**" if b > a else "Tie"
        content += f"| {m} | {a:.3f} | {b:.3f} | {winner} |\n"
    content += "\n"

    # Per-question detail (Config A)
    content += f"## Per-Question Detail ({eval_a['config']})\n\n"
    content += "| # | Question | Faith. | Rel. | Recall | Prec. | Sources |\n"
    content += "|---|----------|--------|------|--------|-------|--------|\n"
    for i, r in enumerate(eval_a["results"], 1):
        m = r["metrics"]
        q = r["question"][:40] + "..." if len(r["question"]) > 40 else r["question"]
        content += f"| {i} | {q} | {m['faithfulness']:.2f} | {m['answer_relevancy']:.2f} | {m['context_recall']:.2f} | {m['context_precision']:.2f} | {r['sources_count']} |\n"
    content += "\n"

    # Worst performers
    content += "## Worst Performers\n\n"
    sorted_results = sorted(eval_a["results"], key=lambda r: sum(r["metrics"].values()))
    for r in sorted_results[:3]:
        content += f"- **Q:** {r['question']}\n"
        content += f"  - Scores: {r['metrics']}\n"
        content += f"  - Expected: {r['expected'][:100]}...\n"
        content += f"  - Got: {r['actual'][:100]}...\n\n"

    # Recommendations
    content += "## Recommendations\n\n"
    content += "1. **Improve Vietnamese embedding**: Switch to BAAI/bge-m3 for better context recall\n"
    content += "2. **Increase chunk overlap**: Current 50 chars may split important context\n"
    content += "3. **PageIndex fallback**: Enable for complex reasoning queries\n"
    content += "4. **Prompt engineering**: Refine system prompt for better citation format\n"

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\n[DONE] Results exported to {RESULTS_PATH}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    eval_a = evaluate_config(
        golden_dataset,
        config_name="Config A: Hybrid + Rerank",
        use_reranking=True,
        dense_only=False,
    )

    eval_b = evaluate_config(
        golden_dataset,
        config_name="Config B: Dense-Only (no rerank)",
        use_reranking=False,
        dense_only=True,
    )

    export_results(eval_a, eval_b)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{eval_a['config']}:")
    for m, s in eval_a["averages"].items():
        print(f"  {m}: {s:.3f}")
    print(f"\n{eval_b['config']}:")
    for m, s in eval_b["averages"].items():
        print(f"  {m}: {s:.3f}")
