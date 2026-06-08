"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import re
import json
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    "https://thanhnien.vn/dien-vien-hai-tran-huu-tin-lanh-7-nam-6-thang-tu-185230428134549434.htm",
    "https://cuoi.tuoitre.vn/loat-nghe-si-viet-tieu-tan-su-nghiep-vi-ma-tuy-20241114142620463.htm",
    "https://vietnamnet.vn/sao-viet-bi-bat-ngoi-tu-mat-danh-tieng-vi-chat-cam-2513746.html",
    "https://thanhnien.vn/nghe-si-dinh-ma-tuy-can-mot-lan-ranh-do-185260520134802695.htm",
    "https://thanhnien.vn/giem-doc-cong-an-hai-phong-chi-dao-dieu-tra-vu-miu-le-duong-tinh-voi-ma-tuy-185260511193728744.htm",
]


class ArticleHTMLParser(HTMLParser):
    """Small dependency-free article parser for Vietnamese news pages."""

    def __init__(self):
        super().__init__()
        self.title = ""
        self.description = ""
        self._current_tag = ""
        self._title_parts = []
        self._paragraphs = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        self._current_tag = tag

        property_name = attrs_dict.get("property") or attrs_dict.get("name")
        content = attrs_dict.get("content", "")
        if property_name in {"og:title", "twitter:title"} and content:
            self.title = content.strip()
        if property_name in {"description", "og:description", "twitter:description"} and content:
            self.description = content.strip()

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        self._current_tag = ""

    def handle_data(self, data):
        if self._skip_depth:
            return

        text = " ".join(data.split())
        if not text:
            return

        if self._current_tag == "title" and not self.title:
            self._title_parts.append(text)
        elif self._current_tag in {"p", "h1", "h2"}:
            self._paragraphs.append(text)

    def get_title(self) -> str:
        title = self.title or " ".join(self._title_parts)
        return unescape(title).strip() or "Unknown"

    def get_markdown(self) -> str:
        pieces = []
        if self.description:
            pieces.append(unescape(self.description).strip())
        pieces.extend(unescape(paragraph).strip() for paragraph in self._paragraphs)
        cleaned = [piece for piece in pieces if len(piece) >= 20]
        return "\n\n".join(dict.fromkeys(cleaned))


def _slug_from_url(url: str, index: int) -> str:
    """Create a stable ASCII filename from URL path."""
    stem = Path(urlparse(url).path).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return f"article_{index:02d}_{stem[:80] or 'news'}.json"


def _parse_article_html(url: str, html: str) -> dict:
    parser = ArticleHTMLParser()
    parser.feed(html)
    content_markdown = parser.get_markdown()
    if len(content_markdown) < 500:
        text = re.sub(r"<[^>]+>", " ", html)
        text = " ".join(unescape(text).split())
        content_markdown = text[:5000]

    return {
        "url": url,
        "title": parser.get_title(),
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_markdown,
    }


async def _crawl_with_requests(url: str) -> dict:
    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return _parse_article_html(url, response.text)


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            title = getattr(result, "metadata", {}).get("title") or "Unknown"
            markdown = getattr(result, "markdown", "") or ""
            if len(markdown) >= 500:
                return {
                    "url": url,
                    "title": title,
                    "date_crawled": datetime.now().isoformat(),
                    "content_markdown": markdown,
                }
    except Exception as exc:
        print(f"  Crawl4AI fallback: {type(exc).__name__}")

    return await _crawl_with_requests(url)


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = _slug_from_url(url, i)
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("Please fill ARTICLE_URLS before running.")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
