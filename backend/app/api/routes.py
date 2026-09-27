"""HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.cost.base import CostEstimator, UnknownTreatmentError
from app.ingestion.pdf_extract import PDFExtractionError
from app.models import (
    AskRequest,
    AskResponse,
    Chunk,
    CostEstimate,
    CoverageRequest,
    CoverageResult,
    Evidence,
    PolicySummary,
    SearchRequest,
    Treatment,
    TreatmentInput,
)
from app.services import PolicyService, compute_coverage

router = APIRouter()


def _policies(request: Request) -> PolicyService:
    return request.app.state.policy_service


def _estimator(request: Request) -> CostEstimator:
    return request.app.state.cost_estimator


def _require_policy(service: PolicyService, doc_id: str) -> None:
    if not service.store.exists(doc_id):
        raise HTTPException(status_code=404, detail=f"Policy '{doc_id}' not found")


class Health(BaseModel):
    status: str
    llm_mode: str
    llm_model: str | None
    retrieval: str
    retrieval_is_semantic: bool
    min_evidence_score: float
    max_query_rewrites: int
    cost_estimator: str


@router.get("/health", response_model=Health)
def health(request: Request) -> Health:
    service = _policies(request)
    return Health(
        status="ok",
        llm_mode="generative" if service.llm else "extractive",
        llm_model=service.llm.model if service.llm else None,
        retrieval=service.retriever.name,
        retrieval_is_semantic=service.retriever.is_semantic,
        min_evidence_score=service.min_evidence_score,
        max_query_rewrites=service.settings.max_query_rewrites,
        cost_estimator="synthetic demo table",
    )


@router.post("/policies", response_model=PolicySummary, status_code=201)
async def upload_policy(request: Request, file: UploadFile = File(...)) -> PolicySummary:
    service = _policies(request)
    limit = service.settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"File exceeds {service.settings.max_upload_mb} MB")
    try:
        return service.ingest(file.filename or "policy.pdf", data)
    except PDFExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/policies", response_model=list[PolicySummary])
def list_policies(request: Request) -> list[PolicySummary]:
    return _policies(request).store.list_summaries()


@router.get("/policies/{doc_id}", response_model=PolicySummary)
def get_policy(request: Request, doc_id: str) -> PolicySummary:
    service = _policies(request)
    _require_policy(service, doc_id)
    return service.store.get_summary(doc_id)


@router.get("/policies/{doc_id}/chunks", response_model=list[Chunk])
def get_chunks(request: Request, doc_id: str, page: int | None = None) -> list[Chunk]:
    service = _policies(request)
    _require_policy(service, doc_id)
    chunks = service.store.get_chunks(doc_id)
    return [c for c in chunks if page is None or c.page == page]


@router.post("/policies/{doc_id}/search", response_model=list[Evidence])
def search_policy(request: Request, doc_id: str, body: SearchRequest) -> list[Evidence]:
    service = _policies(request)
    _require_policy(service, doc_id)
    return [
        Evidence(
            chunk_id=h.chunk.chunk_id, doc_name=h.chunk.doc_name, page=h.chunk.page,
            section=h.chunk.section, text=h.chunk.text, score=h.score,
        )
        for h in service.search(doc_id, body.query, body.top_k)
    ]


@router.post("/policies/{doc_id}/ask", response_model=AskResponse)
def ask_policy(request: Request, doc_id: str, body: AskRequest) -> AskResponse:
    service = _policies(request)
    _require_policy(service, doc_id)
    return service.ask(doc_id, body.question, body.top_k)


@router.get("/treatments", response_model=list[Treatment])
def list_treatments(request: Request) -> list[Treatment]:
    return _estimator(request).list_treatments()


@router.post("/estimate", response_model=CostEstimate)
def estimate_cost(request: Request, body: TreatmentInput) -> CostEstimate:
    try:
        return _estimator(request).estimate(body)
    except UnknownTreatmentError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown treatment_code {exc}") from exc


@router.post("/coverage", response_model=CoverageResult)
def coverage(request: Request, body: CoverageRequest) -> CoverageResult:
    try:
        return compute_coverage(body, _estimator(request))
    except UnknownTreatmentError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown treatment_code {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
