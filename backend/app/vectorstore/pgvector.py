"""PostgreSQL + pgvector vector store.

Searches are always filtered to one document, so an exact cosine scan over that
document's chunks (via the btree index on (embedding_model, doc_id)) is used
instead of an approximate HNSW/IVF index. The `embedding` column is untyped
`vector` so different embedding models (with different dimensions) can coexist;
rows are always filtered by model before distances are computed.
"""

from __future__ import annotations

import numpy as np

from app.models import Chunk
from app.retrieval.base import ScoredChunk
from app.vectorstore.base import check_vectors

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS policy_index (
    embedding_model text NOT NULL,
    doc_id          text NOT NULL,
    chunk_count     integer NOT NULL,
    indexed_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (embedding_model, doc_id)
);

CREATE TABLE IF NOT EXISTS policy_chunks (
    embedding_model text NOT NULL,
    doc_id          text NOT NULL,
    chunk_id        text NOT NULL,
    doc_name        text NOT NULL,
    page            integer NOT NULL,
    section         text,
    text            text NOT NULL,
    char_start      integer NOT NULL,
    char_end        integer NOT NULL,
    embedding       vector NOT NULL,
    PRIMARY KEY (embedding_model, chunk_id)
);

CREATE INDEX IF NOT EXISTS policy_chunks_model_doc_idx ON policy_chunks (embedding_model, doc_id);
"""

_COLUMNS = "chunk_id, doc_id, doc_name, page, section, text, char_start, char_end"


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in np.asarray(vector, dtype=np.float32).ravel()) + "]"


class PgVectorStore:
    name = "pgvector"

    def __init__(self, dsn: str) -> None:
        import psycopg  # optional dependency: pip install -e 'backend[pgvector]'

        self._psycopg = psycopg
        self._dsn = dsn
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def _connect(self):
        return self._psycopg.connect(self._dsn)

    def upsert(self, model: str, doc_id: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = check_vectors(chunks, vectors)
        with self._connect() as conn:  # one transaction: committed on success, rolled back on error
            conn.execute(
                "DELETE FROM policy_chunks WHERE embedding_model = %s AND doc_id = %s", (model, doc_id)
            )
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO policy_chunks (embedding_model, {_COLUMNS}, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)",
                    [
                        (model, c.chunk_id, c.doc_id, c.doc_name, c.page, c.section, c.text,
                         c.char_start, c.char_end, _vector_literal(v))
                        for c, v in zip(chunks, vectors)
                    ],
                )
            conn.execute(
                "INSERT INTO policy_index (embedding_model, doc_id, chunk_count) VALUES (%s, %s, %s) "
                "ON CONFLICT (embedding_model, doc_id) DO UPDATE "
                "SET chunk_count = EXCLUDED.chunk_count, indexed_at = now()",
                (model, doc_id, len(chunks)),
            )

    def has(self, model: str, doc_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM policy_index WHERE embedding_model = %s AND doc_id = %s", (model, doc_id)
            ).fetchone()
        return row is not None

    def query(self, model: str, doc_id: str, vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        if not self.has(model, doc_id):
            raise KeyError(doc_id)
        literal = _vector_literal(vector)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS}, 1 - (embedding <=> %s::vector) AS score FROM policy_chunks "
                "WHERE embedding_model = %s AND doc_id = %s "
                "ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s",
                (literal, model, doc_id, literal, top_k),
            ).fetchall()
        return [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id=r[0], doc_id=r[1], doc_name=r[2], page=r[3], section=r[4], text=r[5],
                    char_start=r[6], char_end=r[7],
                ),
                score=round(float(r[8]), 4),
            )
            for r in rows
        ]
