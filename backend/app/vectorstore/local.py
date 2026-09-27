"""File-based vector store (no database needed). One .npz file per (model, document)."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from app.models import Chunk
from app.retrieval.base import ScoredChunk
from app.vectorstore.base import check_vectors


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model)


class LocalVectorStore:
    name = "local-files"

    def __init__(self, root: Path) -> None:
        self._root = root
        self._cache: dict[Path, tuple[float, np.ndarray, list[Chunk]]] = {}

    def _path(self, model: str, doc_id: str) -> Path:
        if not doc_id.isalnum():
            raise KeyError(doc_id)
        return self._root / _slug(model) / f"{doc_id}.npz"

    def upsert(self, model: str, doc_id: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        path = self._path(model, doc_id)
        vectors = check_vectors(chunks, vectors)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({"model": model, "chunks": [c.model_dump() for c in chunks]})
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez(fh, vectors=vectors, meta=np.array(meta))
            os.replace(tmp, path)  # atomic: readers never see a half-written index
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        self._cache.pop(path, None)

    def has(self, model: str, doc_id: str) -> bool:
        try:
            return self._path(model, doc_id).is_file()
        except KeyError:
            return False

    def _load(self, path: Path) -> tuple[np.ndarray, list[Chunk]]:
        mtime = path.stat().st_mtime
        cached = self._cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1], cached[2]
        with np.load(path, allow_pickle=False) as data:
            vectors = data["vectors"]
            meta = json.loads(str(data["meta"]))
        chunks = [Chunk.model_validate(c) for c in meta["chunks"]]
        self._cache[path] = (mtime, vectors, chunks)
        return vectors, chunks

    def query(self, model: str, doc_id: str, vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        if not self.has(model, doc_id):
            raise KeyError(doc_id)
        vectors, chunks = self._load(self._path(model, doc_id))
        if not chunks:
            return []
        scores = vectors @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [ScoredChunk(chunk=chunks[i], score=round(float(scores[i]), 4)) for i in order]
