"""Page-by-page text extraction from text-based PDFs (no OCR)."""

from __future__ import annotations

import io
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.models import PageText


class PDFExtractionError(ValueError):
    """Raised when a file cannot be read as a text PDF."""


_SPACES = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def _normalise(text: str) -> str:
    lines = [_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def extract_pages(data: bytes) -> list[PageText]:
    """Return the text of every page with its 1-based page number.

    Pages without a text layer (e.g. scans) are returned with empty text and
    `empty_text=True`; their content is never guessed.
    """
    if not data.startswith(b"%PDF"):
        raise PDFExtractionError("File is not a PDF")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise PDFExtractionError("Encrypted PDFs are not supported")
        pages: list[PageText] = []
        for index, page in enumerate(reader.pages, start=1):
            text = _normalise(page.extract_text() or "")
            pages.append(PageText(page=index, text=text, empty_text=not text))
    except PdfReadError as exc:
        raise PDFExtractionError(f"Could not read PDF: {exc}") from exc
    if not pages:
        raise PDFExtractionError("PDF has no pages")
    return pages
