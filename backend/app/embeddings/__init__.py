"""Embedding providers."""

from __future__ import annotations

from app.config import Settings
from app.embeddings.base import Embedder


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider
    if provider == "fastembed":
        from app.embeddings.fastembed_embedder import FastEmbedEmbedder

        return FastEmbedEmbedder(settings.embedding_model, settings.data_dir / "models")
    if provider == "hashing":
        from app.embeddings.hashing import HashingEmbedder

        return HashingEmbedder()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER '{provider}' (use 'fastembed' or 'hashing')")
