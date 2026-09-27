"""Embedding-based retriever over a persistent vector store."""

from __future__ import annotations

import numpy as np

from app.embeddings.base import Embedder
from app.models import Chunk
from app.retrieval.base import NotIndexedError, ScoredChunk
from app.vectorstore.base import VectorStore


def embedding_text(chunk: Chunk) -> str:
    """Text that is embedded for a chunk: its section heading gives the passage context."""
    if chunk.section and not chunk.text.startswith(chunk.section):
        return f"{chunk.section}\n{chunk.text}"
    return chunk.text


class VectorRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store
        self.name = f"{embedder.name} @ {store.name}"
        self.default_min_score = embedder.default_min_score
        self.is_semantic = embedder.is_semantic

    def index(self, doc_id: str, chunks: list[Chunk]) -> None:
        vectors = self.embedder.embed_documents([embedding_text(c) for c in chunks])
        self.store.upsert(self.embedder.name, doc_id, chunks, vectors)

    def is_indexed(self, doc_id: str) -> bool:
        return self.store.has(self.embedder.name, doc_id)

    def search(self, doc_id: str, query: str, top_k: int) -> list[ScoredChunk]:
        if not self.is_indexed(doc_id):
            raise NotIndexedError(doc_id)
        if not query.strip():
            return []
        vector = self.embedder.embed_query(query)
        if not np.any(vector):  # e.g. a query made only of stop words
            return []
        return self.store.query(self.embedder.name, doc_id, vector, top_k)
