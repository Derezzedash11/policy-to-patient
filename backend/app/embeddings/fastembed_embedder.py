"""Local semantic embeddings via fastembed (ONNX runtime, CPU, no API key)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.embeddings.base import l2_normalise


class FastEmbedEmbedder:
    is_semantic = True
    # Uncalibrated starting point for bge-small cosine similarity; tune on an evaluation set.
    default_min_score = 0.55

    def __init__(self, model_name: str, cache_dir: Path | None = None) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
            raise RuntimeError(
                "EMBEDDING_PROVIDER=fastembed needs the optional dependency: "
                "pip install -e 'backend[embeddings]' (or set EMBEDDING_PROVIDER=hashing for "
                "offline, non-semantic testing)"
            ) from exc
        self._model = TextEmbedding(
            model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None
        )
        self.dim = int(self._model.embedding_size)
        self.name = f"fastembed:{model_name}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalise(np.stack(list(self._model.passage_embed(texts))))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalise(next(iter(self._model.query_embed(text))))
