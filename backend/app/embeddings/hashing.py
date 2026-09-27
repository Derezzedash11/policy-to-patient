"""Deterministic feature-hashing embedder.

LEXICAL, NOT SEMANTIC: it matches words and word fragments, not meaning. It exists
for tests and for running the pipeline offline when no embedding model can be
downloaded. Use the fastembed provider for real semantic retrieval.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from app.embeddings.base import l2_normalise

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = frozenset(
    """a about above after again all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers him his how i if in into is it its itself just me more
    most my no nor not of off on once only or other our out over own same she should so some such
    than that the their them then there these they this those through to too under until up very
    was we were what when where which while who whom why will with would you your yours
    policy policies cover covers covered coverage plan""".split()
)
# Weight of each feature family: whole words dominate, fragments give fuzzy matches.
_WORD, _BIGRAM, _CHARGRAM = 1.0, 0.5, 0.25


def _stem(token: str) -> str:
    for suffix in ("ation", "ing", "ies", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _features(text: str) -> list[tuple[str, float]]:
    words = [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP_WORDS]
    feats: list[tuple[str, float]] = [(f"w:{w}", _WORD) for w in words]
    feats += [(f"b:{a}_{b}", _BIGRAM) for a, b in zip(words, words[1:])]
    for w in words:
        padded = f"<{w}>"
        feats += [(f"c:{padded[i:i + 4]}", _CHARGRAM) for i in range(len(padded) - 3)]
    return feats


class HashingEmbedder:
    is_semantic = False
    # Unrelated queries scored <= 0.07 on the test fixture; borderline lexical matches pass to the
    # LLM, which must cite them or answer INSUFFICIENT_EVIDENCE.
    default_min_score = 0.10

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hashing-v1:{dim}"

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feature, weight in _features(text):
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            vec[h % self.dim] += weight if (h >> 63) & 1 else -weight
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalise(np.stack([self._embed(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalise(self._embed(text))
