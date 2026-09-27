from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from pdf_factory import FICTIONAL_POLICY_PAGES, make_text_pdf  # noqa: E402

from app.ingestion.chunker import chunk_pages  # noqa: E402
from app.ingestion.pdf_extract import extract_pages  # noqa: E402
from app.models import Chunk, PageText  # noqa: E402


class StubLLM:
    """Deterministic stand-in for the LLM: returns a canned reply and records prompts."""

    model = "stub-llm"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


@pytest.fixture
def policy_pdf() -> bytes:
    return make_text_pdf(FICTIONAL_POLICY_PAGES)


@pytest.fixture
def policy_pages(policy_pdf: bytes) -> list[PageText]:
    return extract_pages(policy_pdf)


@pytest.fixture
def policy_chunks(policy_pages: list[PageText]) -> list[Chunk]:
    return chunk_pages(policy_pages, doc_id="testdoc", doc_name="fictional_policy.pdf")
