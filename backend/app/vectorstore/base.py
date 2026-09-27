"""Vector store interface: persistent chunk vectors, searched per document."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from app.models import Chunk
from app.retrieval.base import ScoredChunk


class VectorStore(Protocol):
    name: str

    def upsert(self, model: str, doc_id: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Replace all vectors for (model, doc_id) atomically. vectors[i] belongs to chunks[i]."""

    def has(self, model: str, doc_id: str) -> bool:
        """True when (model, doc_id) has been fully indexed (possibly with zero chunks)."""

    def query(self, model: str, doc_id: str, vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        """Top-k chunks of doc_id by cosine similarity, best first. Only that document is searched."""


def check_vectors(chunks: list[Chunk], vectors: np.ndarray) -> np.ndarray:
    """Validate that there is exactly one vector per chunk."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if len(vectors) != len(chunks) or (chunks and vectors.ndim != 2):
        raise ValueError(f"Expected {len(chunks)} vectors, got array of shape {vectors.shape}")
    return vectors
