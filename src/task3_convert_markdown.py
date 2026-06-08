"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import zipfile
from html import unescape
from xml.etree import ElementTree
from pathlib import Path

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _safe_display(text: str) -> str:
    """Render text safely on Windows consoles that are not UTF-8."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _clean_markdown(content: str) -> str:
    """Normalize markdown output for git-friendly generated files."""
    lines = [line.rstrip() for line in content.splitlines()]
    return "\n".join(lines).strip() + "\n"


def _read_docx_text(filepath: Path) -> str:
    """Fallback DOCX reader for environments where MarkItDown is unavailable."""
    with zipfile.ZipFile(filepath) as archive:
        xml_content = archive.read("word/document.xml")

    root = ElementTree.fromstring(xml_content)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [
            node.text
            for node in paragraph.findall(".//w:t", namespace)
            if node.text
        ]
        if texts:
            paragraphs.append("".join(texts))

    return "\n\n".join(paragraphs)


def _convert_document(filepath: Path) -> str:
    """Convert a legal source document to markdown text."""
    if MarkItDown is not None:
        try:
            md = MarkItDown()
            result = md.convert(str(filepath))
            text = getattr(result, "text_content", "") or ""
            if text.strip():
                return text
        except Exception as exc:
            print(f"  MarkItDown fallback: {type(exc).__name__}")

    if filepath.suffix.lower() == ".docx":
        return _read_docx_text(filepath)

    raise RuntimeError(
        f"Cannot convert {filepath.name}. Install markitdown for {filepath.suffix} support."
    )


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = []
    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {_safe_display(filepath.name)}")
            content = _convert_document(filepath)
            output_path = output_dir / f"{filepath.stem}.md"
            header = f"# {filepath.stem}\n\n"
            header += f"**Source:** {filepath.name}\n"
            header += "**Document Type:** legal\n\n---\n\n"
            output_path.write_text(_clean_markdown(header + content), encoding="utf-8")
            converted.append(output_path)
            print(f"  Saved: {_safe_display(str(output_path))}")

    return converted


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = []
    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {_safe_display(filepath.name)}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            title = data.get("title", "Unknown")
            header = f"# {title}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n"
            header += "**Document Type:** news\n\n---\n\n"

            content = data.get("content_markdown") or data.get("content") or ""
            content = unescape(str(content)).strip()
            output_path.write_text(_clean_markdown(header + content), encoding="utf-8")
            converted.append(output_path)
            print(f"  Saved: {_safe_display(str(output_path))}")

    return converted


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\nDone! Output at:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
