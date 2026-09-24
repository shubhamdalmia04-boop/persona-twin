"""Text extraction from Word and PDF files."""
from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def read_docx(path: Path) -> str:
    """Plain text of a .docx (standard library only). Returns '' if unreadable."""
    try:
        with zipfile.ZipFile(path) as z:
            root = ET.fromstring(z.read("word/document.xml"))
    except (OSError, KeyError, zipfile.BadZipFile, ET.ParseError):
        return ""
    paragraphs = ["".join(t.text or "" for t in p.iter(f"{_W}t")) for p in root.iter(f"{_W}p")]
    return "\n\n".join(p for p in paragraphs if p.strip())


def read_pdf(path: Path, max_pages: int = 300) -> str:
    """Plain text of a PDF via pypdf. Returns '' for scanned/encrypted/unreadable files."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if reader.is_encrypted:
            return ""
        pages = [(page.extract_text() or "") for page in reader.pages[:max_pages]]
    except Exception:
        return ""
    return "\n\n".join(p.strip() for p in pages if p.strip())


def read_document(path: Path) -> str:
    return read_docx(path) if path.suffix.lower() == ".docx" else read_pdf(path)
