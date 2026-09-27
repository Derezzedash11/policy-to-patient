import sys
import types

import numpy as np
import pytest

from app.config import Settings
from app.embeddings import build_embedder
from app.embeddings.hashing import HashingEmbedder


def test_hashing_is_deterministic_across_instances():
    a = HashingEmbedder().embed_documents(["Room rent up to Rs 5,000 per day."])
    b = HashingEmbedder().embed_documents(["Room rent up to Rs 5,000 per day."])
    assert np.array_equal(a, b)


def test_hashing_vectors_are_normalised_with_expected_shape():
    vecs = HashingEmbedder(dim=256).embed_documents(["deductible applies", "co-payment of 10 percent"])
    assert vecs.shape == (2, 256) and vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_hashing_empty_inputs():
    emb = HashingEmbedder()
    assert emb.embed_documents([]).shape == (0, emb.dim)
    assert not np.any(emb.embed_query("the and of"))  # only stop words → zero vector


def test_hashing_similarity_prefers_shared_terms():
    emb = HashingEmbedder()
    q = emb.embed_query("co-payment percentage")
    docs = emb.embed_documents(["The insured bears a co-payment of 10 percent.", "Room rent per day."])
    scores = docs @ q
    assert scores[0] > scores[1]


class _FakeTextEmbedding:
    """Mimics fastembed.TextEmbedding without downloading a model."""

    embedding_size = 3
    calls: list[tuple[str, object]] = []

    def __init__(self, model_name, cache_dir=None):
        self.model_name, self.cache_dir = model_name, cache_dir

    def passage_embed(self, texts):
        self.calls.append(("passage", list(texts)))
        return iter([np.array([3.0, 4.0, 0.0]) for _ in texts])

    def query_embed(self, query):
        self.calls.append(("query", query))
        return iter([np.array([0.0, 0.0, 2.0])])


@pytest.fixture
def fake_fastembed(monkeypatch):
    _FakeTextEmbedding.calls = []
    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=_FakeTextEmbedding))


def test_fastembed_adapter_uses_passage_and_query_modes(fake_fastembed, tmp_path):
    emb = build_embedder(Settings(data_dir=tmp_path, embedding_provider="fastembed", embedding_model="m"))
    assert emb.name == "fastembed:m" and emb.dim == 3 and emb.is_semantic
    assert emb._model.cache_dir == str(tmp_path / "models")
    docs = emb.embed_documents(["a", "b"])
    q = emb.embed_query("question")
    assert np.allclose(docs, [[0.6, 0.8, 0.0], [0.6, 0.8, 0.0]])  # normalised
    assert np.allclose(q, [0.0, 0.0, 1.0])
    assert _FakeTextEmbedding.calls == [("passage", ["a", "b"]), ("query", "question")]


def test_build_embedder_rejects_unknown_provider(tmp_path):
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        build_embedder(Settings(data_dir=tmp_path, embedding_provider="nope"))
