"""Retriever interface. Phase 2 can add an embeddings/pgvector implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.models import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def index(self, chunks: list[Chunk]) -> None:
        """(Re)build the index over these chunks."""

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Return up to top_k chunks, best first, with similarity scores in [0, 1]."""
