"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.

Gợi ý nguồn:
    - https://thuvienphapluat.vn
    - https://vanban.chinhphu.vn
    - https://luatvietnam.vn

Gợi ý văn bản:
    - Luật Phòng, chống ma tuý 2021 (73/2021/QH15)
    - Nghị định 105/2021/NĐ-CP
    - Bộ luật Hình sự 2015 (sửa đổi 2017) - Chương XX
    - Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
VALID_EXTENSIONS = {".pdf", ".docx", ".doc"}
MIN_FILE_SIZE_BYTES = 1024


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Directory ready: {DATA_DIR}")


def list_legal_documents() -> list[Path]:
    """Return valid legal document files already collected for Task 1."""
    if not DATA_DIR.exists():
        return []

    return sorted(
        filepath
        for filepath in DATA_DIR.iterdir()
        if filepath.is_file() and filepath.suffix.lower() in VALID_EXTENSIONS
    )


def validate_legal_documents(min_count: int = 3) -> list[dict]:
    """Validate Task 1 inputs and return a small report for demo/tests."""
    setup_directory()
    documents = list_legal_documents()
    report = [
        {
            "filename": path.name,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "is_valid": path.stat().st_size > MIN_FILE_SIZE_BYTES,
        }
        for path in documents
    ]

    valid_count = sum(item["is_valid"] for item in report)
    if valid_count < min_count:
        raise ValueError(
            f"Task 1 needs at least {min_count} non-empty legal documents; "
            f"found {valid_count}."
        )

    return report


# TODO: Tải file PDF/DOCX về DATA_DIR
# Có thể tải thủ công hoặc viết script download nếu có direct link.
#
# Ví dụ nếu có direct link:
#
# import requests
#
# def download_file(url: str, filename: str):
#     response = requests.get(url)
#     filepath = DATA_DIR / filename
#     filepath.write_bytes(response.content)
#     print(f"✓ Đã tải: {filepath}")


if __name__ == "__main__":
    for item in validate_legal_documents():
        status = "OK" if item["is_valid"] else "TOO_SMALL"
        filename = item["filename"].encode("ascii", "backslashreplace").decode("ascii")
        print(f"{status}: {filename} ({item['size_bytes']} bytes)")
