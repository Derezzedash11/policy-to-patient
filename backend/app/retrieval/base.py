"""Retriever interface: index a document once, then search it by doc_id."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.models import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class NotIndexedError(KeyError):
    """The document has no index for the current retriever configuration."""


class Retriever(Protocol):
    name: str
    default_min_score: float
    is_semantic: bool

    def index(self, doc_id: str, chunks: list[Chunk]) -> None:
        """Build and persist the index for one document (replacing any previous one)."""

    def is_indexed(self, doc_id: str) -> bool:
        """True when doc_id can be searched without re-indexing."""

    def search(self, doc_id: str, query: str, top_k: int) -> list[ScoredChunk]:
        """Up to top_k chunks of doc_id only, best first. Raises NotIndexedError if not indexed."""
