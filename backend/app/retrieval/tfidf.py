"""Lexical TF-IDF retriever (Phase 1 baseline)."""

from __future__ import annotations

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from app.models import Chunk
from app.retrieval.base import ScoredChunk

# Words that appear in almost every policy question but say nothing about the topic.
# Without this, "Does the policy cover X?" matches any passage containing "policy".
DOMAIN_STOP_WORDS = frozenset({"policy", "policies", "cover", "covers", "covered", "coverage", "plan"})
_STOP_WORDS = sorted(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS)


class TfidfRetriever:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        if not self._chunks:
            self._vectorizer, self._matrix = None, None
            return
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=_STOP_WORDS,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        # Section headings are indexed with the text so heading words help ranking.
        docs = [f"{c.section or ''}\n{c.text}" for c in self._chunks]
        try:
            self._matrix = self._vectorizer.fit_transform(docs)
        except ValueError:  # vocabulary empty (e.g. only stop words)
            self._vectorizer, self._matrix = None, None

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if self._vectorizer is None or self._matrix is None or not query.strip():
            return []
        query_vec = self._vectorizer.transform([query])
        scores = linear_kernel(query_vec, self._matrix).ravel()  # TF-IDF rows are L2-normalised
        order = scores.argsort()[::-1][:top_k]
        return [
            ScoredChunk(chunk=self._chunks[i], score=round(float(scores[i]), 4))
            for i in order
            if scores[i] > 0
        ]
