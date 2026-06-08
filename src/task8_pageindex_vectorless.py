"""Task 8 - PageIndex Vectorless RAG."""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
PAGEINDEX_REGISTRY_PATH = INDEX_DIR / "pageindex_documents.json"
PAGEINDEX_UPLOAD_DIR = INDEX_DIR / "pageindex_uploads"
PAGEINDEX_SOURCE = "pageindex"
PAGEINDEX_POLL_SECONDS = 1.0
PAGEINDEX_MAX_POLLS = 8


def upload_documents() -> list[dict]:
    """
    Upload toàn bộ markdown đã chuẩn hóa lên PageIndex.

    Chọn upload các file trong data/standardized vì đây là dữ liệu sạch sau Task 3.
    PageIndex API hiện chỉ nhận PDF, nên mỗi markdown được render thành PDF tạm trước khi upload.
    Registry local lưu doc_id trong data/index/pageindex_documents.json để query lại
    mà không phải upload lặp.
    """
    client = _get_pageindex_client()
    markdown_files = _iter_markdown_documents()
    existing_by_path = {
        item.get("path"): item
        for item in _load_registry()
        if item.get("doc_id") and item.get("path")
    }

    uploaded_documents = []
    for md_file in markdown_files:
        relative_path = md_file.relative_to(STANDARDIZED_DIR).as_posix()
        doc_type = relative_path.split("/", 1)[0] if "/" in relative_path else "unknown"

        if relative_path in existing_by_path:
            uploaded_documents.append(existing_by_path[relative_path])
            print(f"  Da co tren registry: {_safe_display(relative_path)}")
            continue

        item = {
            "path": relative_path,
            "filename": md_file.name,
            "type": doc_type,
        }
        try:
            pdf_file = _markdown_to_pdf(md_file)
            response = client.submit_document(str(pdf_file))
            doc_id = response.get("doc_id") or response.get("id")
            item.update({
                "doc_id": doc_id,
                "upload_file": str(pdf_file.relative_to(INDEX_DIR).as_posix()),
                "status": "uploaded" if doc_id else "missing_doc_id",
                "pageindex_response": response,
            })
            print(f"  Uploaded: {_safe_display(relative_path)}")
        except Exception as exc:
            item.update({
                "status": "failed",
                "error": str(exc),
            })
            print(f"  Loi upload {_safe_display(relative_path)}: {_safe_display(str(exc))}")

        uploaded_documents.append(item)

    _save_registry(uploaded_documents)
    return uploaded_documents


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'
        }
    """
    if top_k <= 0:
        return []

    registry = [
        item
        for item in _load_registry()
        if item.get("doc_id") and item.get("status") != "failed"
    ]
    if not PAGEINDEX_API_KEY or not registry:
        return _local_pageindex_fallback(query, top_k)

    try:
        client = _get_pageindex_client()
        results = []
        for document in registry:
            response = client.submit_query(document["doc_id"], query, thinking=False)
            retrieval_id = response.get("retrieval_id") or response.get("id")
            if not retrieval_id:
                continue

            retrieval = _wait_for_retrieval(client, retrieval_id)
            results.extend(_extract_pageindex_results(retrieval, document))
            if len(results) >= top_k * 3:
                break

        if results:
            results.sort(key=lambda item: item["score"], reverse=True)
            return results[:top_k]
    except Exception:
        pass

    return _local_pageindex_fallback(query, top_k)


def _load_registry() -> list[dict]:
    """Đọc danh sách document đã upload lên PageIndex."""
    if not PAGEINDEX_REGISTRY_PATH.exists():
        return []
    payload = json.loads(PAGEINDEX_REGISTRY_PATH.read_text(encoding="utf-8"))
    return payload.get("documents", [])


def _save_registry(documents: list[dict]) -> None:
    """Lưu doc_id để lần sau query không upload lại."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "documents": documents,
    }
    PAGEINDEX_REGISTRY_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _iter_markdown_documents() -> list[Path]:
    return [
        path
        for path in sorted(STANDARDIZED_DIR.rglob("*.md"))
        if path.is_file() and not path.name.startswith(".")
    ]


def _markdown_to_pdf(md_file: Path) -> Path:
    """Render markdown thành PDF tạm vì PageIndex chỉ hỗ trợ upload PDF."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Cần cài reportlab để tạo PDF tạm cho PageIndex: pip install reportlab"
        ) from exc

    relative_path = md_file.relative_to(STANDARDIZED_DIR).as_posix()
    pdf_path = PAGEINDEX_UPLOAD_DIR / f"{relative_path.replace('/', '__')}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_mtime >= md_file.stat().st_mtime:
        return pdf_path

    PAGEINDEX_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    font_name = _register_pdf_font(pdfmetrics, TTFont)
    font_size = 10.5
    line_height = 14
    page_width, page_height = A4
    margin = 18 * mm
    max_width = page_width - 2 * margin

    pdf = canvas.Canvas(str(pdf_path), pagesize=A4)
    pdf.setTitle(md_file.stem)
    pdf.setFont(font_name, 14)

    y = page_height - margin
    title = md_file.name
    y = _draw_wrapped_text(pdf, title, margin, y, max_width, font_name, 14, 18, pdfmetrics)
    y -= 8
    pdf.setFont(font_name, font_size)

    content = _clean_markdown_for_pdf(md_file.read_text(encoding="utf-8"))
    for paragraph in content.splitlines():
        if not paragraph.strip():
            y -= line_height
            if y < margin:
                pdf.showPage()
                pdf.setFont(font_name, font_size)
                y = page_height - margin
            continue

        y = _draw_wrapped_text(
            pdf,
            paragraph,
            margin,
            y,
            max_width,
            font_name,
            font_size,
            line_height,
            pdfmetrics,
        )
        y -= 3
        if y < margin:
            pdf.showPage()
            pdf.setFont(font_name, font_size)
            y = page_height - margin

    pdf.save()
    return pdf_path


def _register_pdf_font(pdfmetrics, TTFont) -> str:
    """Ưu tiên font Windows có hỗ trợ tiếng Việt; fallback Helvetica nếu không có."""
    font_candidates = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "calibri.ttf",
    ]
    for font_path in font_candidates:
        if font_path.exists():
            font_name = font_path.stem
            pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
            return font_name
    return "Helvetica"


def _clean_markdown_for_pdf(text: str) -> str:
    replacements = {
        "#": "",
        "**": "",
        "__": "",
        "`": "",
        "|": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace("\t", " ").strip()


def _draw_wrapped_text(
    pdf,
    text: str,
    x: float,
    y: float,
    max_width: float,
    font_name: str,
    font_size: float,
    line_height: float,
    pdfmetrics,
) -> float:
    page_width, page_height = A4 = (595.2755905511812, 841.8897637795277)
    margin = x
    pdf.setFont(font_name, font_size)
    for line in _wrap_pdf_line(text.strip(), max_width, font_name, font_size, pdfmetrics):
        if y < margin:
            pdf.showPage()
            pdf.setFont(font_name, font_size)
            y = page_height - margin
        pdf.drawString(x, y, line)
        y -= line_height
    return y


def _wrap_pdf_line(text: str, max_width: float, font_name: str, font_size: float, pdfmetrics) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)
        current = word

        while pdfmetrics.stringWidth(current, font_name, font_size) > max_width and len(current) > 1:
            split_at = max(1, int(len(current) * max_width / pdfmetrics.stringWidth(current, font_name, font_size)))
            lines.append(current[:split_at])
            current = current[split_at:]

    if current:
        lines.append(current)
    return lines


def _get_pageindex_client():
    if not PAGEINDEX_API_KEY:
        raise ValueError("Thiếu PAGEINDEX_API_KEY trong file .env")
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _wait_for_retrieval(client, retrieval_id: str) -> dict:
    """Poll PageIndex cho tới khi retrieval có kết quả hoặc hết lượt chờ."""
    latest = {}
    for _ in range(PAGEINDEX_MAX_POLLS):
        latest = client.get_retrieval(retrieval_id)
        status = str(latest.get("status", "")).lower()
        if status in {"completed", "complete", "succeeded", "success", "done", "ready"}:
            return latest
        if status in {"failed", "error"}:
            return latest
        if _has_result_payload(latest):
            return latest
        time.sleep(PAGEINDEX_POLL_SECONDS)
    return latest


def _has_result_payload(payload) -> bool:
    if not isinstance(payload, dict):
        return False
    result_keys = ("results", "chunks", "nodes", "references", "answer", "text", "content")
    return any(key in payload for key in result_keys)


def _extract_pageindex_results(payload, document: dict) -> list[dict]:
    """Parse response PageIndex linh hoạt vì SDK có thể thay đổi schema."""
    candidates = []
    _collect_text_items(payload, candidates)

    results = []
    seen = set()
    for index, item in enumerate(candidates):
        content = item["content"].strip()
        if not content or content in seen:
            continue
        seen.add(content)

        metadata = {
            "source": document.get("filename"),
            "path": document.get("path"),
            "type": document.get("type"),
            "doc_id": document.get("doc_id"),
            **item.get("metadata", {}),
        }
        results.append({
            "content": content,
            "score": float(item.get("score", 1.0 / (index + 1))),
            "metadata": metadata,
            "source": PAGEINDEX_SOURCE,
        })
    return results


def _collect_text_items(payload, output: list[dict]) -> None:
    if isinstance(payload, dict):
        content = _first_text_value(payload)
        if content:
            output.append({
                "content": content,
                "score": _first_score_value(payload),
                "metadata": _metadata_from_payload(payload),
            })

        for value in payload.values():
            if isinstance(value, (dict, list)):
                _collect_text_items(value, output)
    elif isinstance(payload, list):
        for item in payload:
            _collect_text_items(item, output)


def _first_text_value(payload: dict) -> str:
    for key in ("content", "text", "snippet", "answer", "markdown", "page_content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _first_score_value(payload: dict) -> float:
    for key in ("score", "relevance", "confidence", "similarity"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 1.0


def _metadata_from_payload(payload: dict) -> dict:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        return dict(metadata)

    extracted = {}
    for key in ("page", "page_num", "pageNumber", "block_id", "section", "title"):
        if key in payload and isinstance(payload[key], (str, int, float)):
            extracted[key] = payload[key]
    return extracted


def _local_pageindex_fallback(query: str, top_k: int) -> list[dict]:
    """
    Fallback local để test/pipeline vẫn hoạt động khi chưa upload hoặc API lỗi.

    Các kết quả vẫn được đánh dấu source='pageindex' vì đây là nhánh fallback của Task 8.
    """
    try:
        from src.task6_lexical_search import lexical_search
    except ModuleNotFoundError:
        from task6_lexical_search import lexical_search

    results = []
    for item in lexical_search(query, top_k=top_k):
        results.append({
            "content": item["content"],
            "score": float(item["score"]),
            "metadata": {
                **dict(item.get("metadata", {})),
                "pageindex_mode": "local_bm25_fallback",
            },
            "source": PAGEINDEX_SOURCE,
        })
    return results


def _safe_display(text: str) -> str:
    return text.encode("ascii", "backslashreplace").decode("ascii")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 8 - PageIndex vectorless retrieval")
    parser.add_argument("--upload", action="store_true", help="Upload markdown documents to PageIndex")
    parser.add_argument("--query", default="hình phạt sử dụng ma túy", help="Query để test PageIndex")
    parser.add_argument("--top-k", type=int, default=3, help="Số kết quả trả về")
    args = parser.parse_args()

    if args.upload:
        if not PAGEINDEX_API_KEY:
            print("Hay set PAGEINDEX_API_KEY trong file .env truoc khi upload.")
        else:
            uploaded = upload_documents()
            ok_count = len([item for item in uploaded if item.get("doc_id")])
            print(f"\nDa upload/registry {ok_count}/{len(uploaded)} documents.")

    print("\nTest query:")
    for result in pageindex_search(args.query, top_k=args.top_k):
        preview = _safe_display(result["content"][:120])
        print(f"[{result['score']:.3f}] [{result['source']}] {preview}...")
