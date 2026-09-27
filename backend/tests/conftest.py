from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from pdf_factory import FICTIONAL_POLICY_PAGES, make_text_pdf  # noqa: E402

from app.embeddings.base import l2_normalise  # noqa: E402
from app.embeddings.hashing import HashingEmbedder  # noqa: E402
from app.ingestion.chunker import chunk_pages  # noqa: E402
from app.ingestion.pdf_extract import extract_pages  # noqa: E402
from app.models import Chunk, PageText  # noqa: E402
from app.retrieval.vector import VectorRetriever  # noqa: E402
from app.vectorstore.local import LocalVectorStore  # noqa: E402


class StubLLM:
    """Deterministic stand-in for the LLM: returns a canned reply and records prompts."""

    model = "stub-llm"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


class ScriptedLLM:
    """Returns scripted replies in order (e.g. a query rewrite, then an answer) and records calls."""

    model = "scripted-llm"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.replies:
            raise AssertionError("ScriptedLLM called more times than scripted")
        return self.replies.pop(0)


@pytest.fixture
def policy_pdf() -> bytes:
    return make_text_pdf(FICTIONAL_POLICY_PAGES)


@pytest.fixture
def policy_pages(policy_pdf: bytes) -> list[PageText]:
    return extract_pages(policy_pdf)


@pytest.fixture
def policy_chunks(policy_pages: list[PageText]) -> list[Chunk]:
    return chunk_pages(policy_pages, doc_id="testdoc", doc_name="fictional_policy.pdf")


# --------------------------------------------------------------------------- embedders for tests


class ConceptEmbedder:
    """Deterministic stand-in for a semantic model: maps paraphrases to shared concept axes.

    It lets tests check that retrieval works on meaning (vector similarity) rather than shared
    words, without downloading a model. It says nothing about any real model's accuracy.
    """

    name = "concept-test:7"
    dim = 7
    default_min_score = 0.5
    is_semantic = True
    CONCEPTS = [
        ("deductible", ["deductible", "pay myself before", "own pocket before"]),
        ("copay", ["co-payment", "copay", "share of each claim"]),
        ("room", ["room rent", "hospital room", "bed charges"]),
        ("waiting", ["waiting period", "how long must i wait"]),
        ("maternity", ["maternity", "pregnancy", "childbirth"]),
        ("exclusion", ["excluded", "never pays for"]),
        ("limit", ["sum insured", "maximum amount"]),
    ]

    def _embed(self, text: str) -> np.ndarray:
        text = text.lower()
        return np.array(
            [any(p in text for p in phrases) for _, phrases in self.CONCEPTS], dtype=np.float32
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return l2_normalise(np.stack([self._embed(t) for t in texts])) if texts else np.zeros((0, 7))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalise(self._embed(text))


class CountingEmbedder:
    """Wraps an embedder and counts how many documents it embeds."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name, self.dim = inner.name, inner.dim
        self.default_min_score, self.is_semantic = inner.default_min_score, inner.is_semantic
        self.documents_embedded = 0

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.documents_embedded += len(texts)
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self.inner.embed_query(text)


@pytest.fixture
def make_retriever(tmp_path):
    def factory(embedder=None, root=None) -> VectorRetriever:
        return VectorRetriever(embedder or HashingEmbedder(), LocalVectorStore(root or tmp_path / "vectors"))

    return factory


@pytest.fixture
def indexed_retriever(make_retriever, policy_chunks) -> VectorRetriever:
    retriever = make_retriever()
    retriever.index("testdoc", policy_chunks)
    return retriever
