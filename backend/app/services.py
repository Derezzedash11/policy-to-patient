"""Wires storage, ingestion, retrieval, Q&A, cost and rules together for the API."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from app.config import Settings
from app.cost.base import CostEstimator
from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_extract import extract_pages
from app.models import (
    AskResponse,
    CostEstimate,
    CoverageRequest,
    CoverageResult,
    PolicySummary,
)
from app.qa.answer import answer_question
from app.qa.llm import LLMClient
from app.retrieval.base import Retriever, ScoredChunk
from app.retrieval.tfidf import TfidfRetriever
from app.rules.coverage import calculate_coverage
from app.store import PolicyStore


class PolicyService:
    def __init__(
        self,
        settings: Settings,
        store: PolicyStore,
        llm: LLMClient | None,
        retriever_factory: Callable[[], Retriever] = TfidfRetriever,
    ) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm
        self._retriever_factory = retriever_factory
        self._retrievers: dict[str, Retriever] = {}

    def ingest(self, filename: str, data: bytes) -> PolicySummary:
        doc_id = hashlib.sha256(data).hexdigest()[:16]
        pages = extract_pages(data)
        chunks = chunk_pages(pages, doc_id=doc_id, doc_name=filename)
        sections = list(dict.fromkeys(c.section for c in chunks if c.section))
        summary = PolicySummary(
            doc_id=doc_id,
            doc_name=filename,
            page_count=len(pages),
            chunk_count=len(chunks),
            empty_pages=[p.page for p in pages if p.empty_text],
            sections=sections,
        )
        self.store.save(summary, data, pages, chunks)
        self._retrievers.pop(doc_id, None)
        return summary

    def retriever(self, doc_id: str) -> Retriever:
        if doc_id not in self._retrievers:
            retriever = self._retriever_factory()
            retriever.index(self.store.get_chunks(doc_id))  # raises KeyError if unknown
            self._retrievers[doc_id] = retriever
        return self._retrievers[doc_id]

    def search(self, doc_id: str, query: str, top_k: int | None) -> list[ScoredChunk]:
        return self.retriever(doc_id).search(query, top_k or self.settings.retrieval_top_k)

    def ask(self, doc_id: str, question: str, top_k: int | None) -> AskResponse:
        return answer_question(
            question,
            self.retriever(doc_id),
            self.llm,
            top_k=top_k or self.settings.retrieval_top_k,
            min_score=self.settings.min_evidence_score,
        )


def compute_coverage(request: CoverageRequest, estimator: CostEstimator) -> CoverageResult:
    """Resolve the cost (estimator or explicit total) and run the rules engine."""
    estimate: CostEstimate | None = None
    if request.treatment is not None and request.total_cost is not None:
        raise ValueError("Provide either 'treatment' or 'total_cost', not both")
    if request.treatment is not None:
        estimate = estimator.estimate(request.treatment)
        total = estimate.estimate
        category = request.category or estimate.category
        room_rent = request.room_rent_per_day if request.room_rent_per_day is not None else estimate.room_rent_per_day
        los = request.length_of_stay_days if request.length_of_stay_days is not None else estimate.length_of_stay_days
    elif request.total_cost is not None:
        total = request.total_cost
        category, room_rent, los = request.category, request.room_rent_per_day, request.length_of_stay_days
    else:
        raise ValueError("Provide either 'treatment' or 'total_cost'")

    result = calculate_coverage(total, request.terms, category, room_rent, los)
    if estimate is not None:
        result.cost_estimate = estimate
        result.assumptions.append(
            "Total cost is the midpoint of a SYNTHETIC demo estimate, not a real price."
        )
    return result
