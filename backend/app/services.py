"""Wires storage, ingestion, retrieval, Q&A, cost and rules together for the API."""

from __future__ import annotations

import hashlib
import logging

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
from app.rules.coverage import calculate_coverage
from app.store import PolicyStore

logger = logging.getLogger(__name__)


class PolicyService:
    def __init__(
        self,
        settings: Settings,
        store: PolicyStore,
        llm: LLMClient | None,
        retriever: Retriever,
    ) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm
        self.retriever = retriever

    @property
    def min_evidence_score(self) -> float:
        if self.settings.min_evidence_score is not None:
            return self.settings.min_evidence_score
        return self.retriever.default_min_score

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
        # Embed once at ingestion; the summary is written last so a failed index leaves no policy.
        self.retriever.index(doc_id, chunks)
        self.store.save(summary, data, pages, chunks)
        return summary

    def ensure_indexed(self, doc_id: str) -> None:
        """Index from stored chunks only if no index exists for the current embedding model
        (e.g. EMBEDDING_PROVIDER/MODEL changed). A normal restart never re-embeds."""
        if not self.retriever.is_indexed(doc_id):
            logger.warning("No index for %s with %s; indexing stored chunks", doc_id, self.retriever.name)
            self.retriever.index(doc_id, self.store.get_chunks(doc_id))

    def search(self, doc_id: str, query: str, top_k: int | None) -> list[ScoredChunk]:
        self.ensure_indexed(doc_id)
        return self.retriever.search(doc_id, query, top_k or self.settings.retrieval_top_k)

    def ask(self, doc_id: str, question: str, top_k: int | None) -> AskResponse:
        self.ensure_indexed(doc_id)
        return answer_question(
            question,
            doc_id,
            self.retriever,
            self.llm,
            top_k=top_k or self.settings.retrieval_top_k,
            min_score=self.min_evidence_score,
            max_rewrites=self.settings.max_query_rewrites,
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
