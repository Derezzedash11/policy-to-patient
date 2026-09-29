"""Embedder interface."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    """Stable identifier, stored with every vector (e.g. 'fastembed:BAAI/bge-small-en-v1.5')."""
    dim: int
    default_min_score: float
    """Cosine similarity below which a passage is not treated as evidence (uncalibrated default)."""
    is_semantic: bool

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""

    def embed_query(self, text: str) -> np.ndarray:
        """Return a (dim,) float32 L2-normalised vector."""


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)
