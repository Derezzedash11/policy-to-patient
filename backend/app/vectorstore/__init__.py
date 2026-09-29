"""Vector stores."""

from __future__ import annotations

from app.config import Settings
from app.vectorstore.base import VectorStore


def build_vector_store(settings: Settings) -> VectorStore:
    if settings.database_url:
        from app.vectorstore.pgvector import PgVectorStore

        return PgVectorStore(settings.database_url)
    from app.vectorstore.local import LocalVectorStore

    return LocalVectorStore(settings.data_dir / "vectors")
