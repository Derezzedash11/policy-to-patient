"""Contract tests every VectorStore implementation must pass.

The pgvector variant runs only when TEST_DATABASE_URL points at a PostgreSQL database with the
pgvector extension available (its tables are truncated), e.g.
TEST_DATABASE_URL=postgresql://p2p:p2p@localhost:5432/p2p_test
"""

import os

import numpy as np
import pytest

from app.models import Chunk
from app.vectorstore.local import LocalVectorStore

MODEL = "unit-test-model:3"
PG_URL = os.environ.get("TEST_DATABASE_URL")


def _chunk(doc_id: str, i: int, page: int = 1, section: str | None = "2. LIMITS") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:p{page}:c{i}", doc_id=doc_id, doc_name=f"{doc_id}.pdf", page=page,
        section=section, text=f"chunk {i} of {doc_id}", char_start=i * 10, char_end=i * 10 + 9,
    )


def _unit(*xs: float) -> np.ndarray:
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def _make_local(tmp_path):
    return lambda: LocalVectorStore(tmp_path / "vectors")


def _make_pg(tmp_path):
    from app.vectorstore.pgvector import PgVectorStore

    store = PgVectorStore(PG_URL)
    with store._connect() as conn:
        conn.execute("TRUNCATE policy_chunks, policy_index")
    return lambda: PgVectorStore(PG_URL)


@pytest.fixture(
    params=[
        "local",
        pytest.param("pgvector", marks=pytest.mark.skipif(not PG_URL, reason="TEST_DATABASE_URL not set")),
    ]
)
def store_factory(request, tmp_path):
    """Returns a zero-arg factory so tests can simulate an application restart."""
    return _make_local(tmp_path) if request.param == "local" else _make_pg(tmp_path)


def test_upsert_has_and_query_order(store_factory):
    store = store_factory()
    chunks = [_chunk("doca", 0), _chunk("doca", 1, page=2, section=None), _chunk("doca", 2)]
    store.upsert(MODEL, "doca", chunks, np.stack([_unit(1, 0, 0), _unit(0, 1, 0), _unit(1, 1, 0)]))
    assert store.has(MODEL, "doca")
    hits = store.query(MODEL, "doca", _unit(1, 0.1, 0), top_k=2)
    assert [h.chunk.chunk_id for h in hits] == ["doca:p1:c0", "doca:p1:c2"]
    assert hits[0].score > hits[1].score
    assert hits[0].score == pytest.approx(float(_unit(1, 0.1, 0) @ _unit(1, 0, 0)), abs=1e-3)


def test_metadata_round_trips(store_factory):
    store = store_factory()
    chunk = _chunk("docm", 4, page=7, section=None)
    store.upsert(MODEL, "docm", [chunk], _unit(0, 0, 1)[None, :])
    assert store.query(MODEL, "docm", _unit(0, 0, 1), top_k=1)[0].chunk == chunk


def test_persists_across_restart(store_factory):
    first = store_factory()
    first.upsert(MODEL, "docp", [_chunk("docp", 0)], _unit(1, 0, 0)[None, :])
    restarted = store_factory()
    assert restarted.has(MODEL, "docp")
    assert restarted.query(MODEL, "docp", _unit(1, 0, 0), 1)[0].chunk.chunk_id == "docp:p1:c0"


def test_documents_and_models_are_isolated(store_factory):
    store = store_factory()
    store.upsert(MODEL, "doca", [_chunk("doca", 0)], _unit(1, 0, 0)[None, :])
    store.upsert(MODEL, "docb", [_chunk("docb", 0)], _unit(1, 0, 0)[None, :])
    store.upsert("other-model:2", "doca", [_chunk("doca", 9)], _unit(1, 0)[None, :])
    assert [h.chunk.doc_id for h in store.query(MODEL, "doca", _unit(1, 0, 0), 10)] == ["doca"]
    assert [h.chunk.chunk_id for h in store.query("other-model:2", "doca", _unit(1, 0), 10)] == ["doca:p1:c9"]
    assert not store.has("never-used", "doca")


def test_upsert_replaces_previous_vectors(store_factory):
    store = store_factory()
    store.upsert(MODEL, "docr", [_chunk("docr", 0), _chunk("docr", 1)], np.stack([_unit(1, 0, 0)] * 2))
    store.upsert(MODEL, "docr", [_chunk("docr", 5)], _unit(1, 0, 0)[None, :])
    assert [h.chunk.chunk_id for h in store.query(MODEL, "docr", _unit(1, 0, 0), 10)] == ["docr:p1:c5"]


def test_document_with_no_chunks_is_indexed_but_empty(store_factory):
    store = store_factory()
    store.upsert(MODEL, "docempty", [], np.zeros((0, 3), dtype=np.float32))
    assert store.has(MODEL, "docempty")
    assert store.query(MODEL, "docempty", _unit(1, 0, 0), 5) == []


def test_query_unknown_document_raises(store_factory):
    with pytest.raises(KeyError):
        store_factory().query(MODEL, "missing", _unit(1, 0, 0), 5)


def test_local_store_rejects_unsafe_ids(tmp_path):
    store = LocalVectorStore(tmp_path)
    assert not store.has(MODEL, "../escape")
    with pytest.raises(KeyError):
        store.upsert(MODEL, "../escape", [], np.zeros((0, 3)))


def test_vector_count_must_match_chunks(store_factory):
    with pytest.raises(ValueError, match="Expected 2 vectors"):
        store_factory().upsert(MODEL, "docbad", [_chunk("docbad", 0), _chunk("docbad", 1)],
                               _unit(1, 0, 0)[None, :])
