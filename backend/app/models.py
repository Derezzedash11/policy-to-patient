"""Shared data models (API schemas and internal records)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- policy / evidence


class PageText(BaseModel):
    page: int = Field(description="1-based page number in the source PDF")
    text: str
    empty_text: bool = Field(description="True when the page has no extractable text layer")


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    section: str | None = None
    text: str
    char_start: int = Field(description="Offset of the chunk within its page text")
    char_end: int


class PolicySummary(BaseModel):
    doc_id: str
    doc_name: str
    page_count: int
    chunk_count: int
    empty_pages: list[int]
    sections: list[str]


class Evidence(BaseModel):
    chunk_id: str
    doc_name: str
    page: int
    section: str | None
    text: str
    score: float


class Citation(BaseModel):
    ref: str = Field(description="Passage label used in the answer, e.g. C1")
    chunk_id: str
    doc_name: str
    page: int
    section: str | None
    quote: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int | None = Field(default=None, ge=1, le=20)


AnswerStatus = Literal[
    "answered", "evidence_only", "insufficient_evidence", "manual_review", "llm_error"
]


class RetrievalAttempt(BaseModel):
    """One retrieval pass. Attempts after the first come from the self-correcting step."""

    query: str
    strategy: Literal["original", "llm_rewrite"]
    best_score: float | None = None
    evidence_found: bool = False
    note: str | None = None


class AskResponse(BaseModel):
    status: AnswerStatus
    mode: Literal["generative", "extractive"]
    question: str
    answer: str | None = Field(
        default=None, description="LLM explanation of the evidence; null when not generated"
    )
    citations: list[Citation] = []
    evidence: list[Evidence] = []
    retrieval_attempts: list[RetrievalAttempt] = []
    verification_issues: list[str] = Field(
        default=[], description="Why a generated answer was not accepted (status manual_review)"
    )
    message: str


# --------------------------------------------------------------------------- cost estimate

CityTier = Literal["tier1", "tier2", "tier3"]
RoomType = Literal["general_ward", "semi_private", "private"]


class TreatmentInput(BaseModel):
    treatment_code: str
    city_tier: CityTier = "tier1"
    room_type: RoomType = "semi_private"
    length_of_stay_days: int | None = Field(
        default=None, ge=0, le=90, description="Defaults to the treatment's typical stay"
    )


class Treatment(BaseModel):
    treatment_code: str
    treatment_name: str
    category: str
    typical_length_of_stay_days: int


class CostComponent(BaseModel):
    name: str
    low: float
    high: float


class CostEstimate(BaseModel):
    treatment_code: str
    treatment_name: str
    category: str
    city_tier: CityTier
    room_type: RoomType
    length_of_stay_days: int
    currency: str
    low: float
    high: float
    estimate: float = Field(description="Midpoint of the low/high range")
    room_rent_per_day: float = Field(description="Midpoint daily room charge used in the estimate")
    components: list[CostComponent]
    method: str
    is_synthetic: bool
    source: str
    disclaimer: str


# --------------------------------------------------------------------------- coverage rules


class TermCitation(BaseModel):
    """Where a policy term value came from (a chunk returned by retrieval)."""

    chunk_id: str | None = None
    page: int | None = None
    section: str | None = None
    quote: str | None = None


class PolicyTerms(BaseModel):
    """Structured policy terms. Every field is optional; missing terms are reported, never guessed."""

    sum_insured: float | None = Field(default=None, ge=0)
    sum_insured_already_used: float = Field(default=0, ge=0)
    deductible: float | None = Field(default=None, ge=0)
    deductible_already_met: float = Field(default=0, ge=0)
    copay_pct: float | None = Field(default=None, ge=0, le=100)
    room_rent_limit_per_day: float | None = Field(default=None, ge=0)
    sub_limits: dict[str, float] = Field(
        default_factory=dict, description="Per-category maximum payable, e.g. {'cataract': 40000}"
    )
    waiting_period_months: dict[str, int] = Field(
        default_factory=dict, description="Per-category waiting period in months"
    )
    policy_age_months: int | None = Field(default=None, ge=0)
    excluded_categories: list[str] = Field(default_factory=list)
    citations: dict[str, TermCitation] = Field(
        default_factory=dict, description="Evidence for each term, keyed by field name"
    )


class CoverageRequest(BaseModel):
    """Supply exactly one of `treatment` (cost comes from the estimator) or an explicit `total_cost`."""

    terms: PolicyTerms
    treatment: TreatmentInput | None = None
    total_cost: float | None = Field(default=None, ge=0)
    category: str | None = None
    room_rent_per_day: float | None = Field(default=None, ge=0)
    length_of_stay_days: int | None = Field(default=None, ge=0)


class CalculationStep(BaseModel):
    step: int
    rule: str
    description: str
    amount_before: float
    amount_after: float
    patient_share_added: float
    citation: TermCitation | None = None


CoverageStatus = Literal["complete", "incomplete_terms", "not_covered"]


class CoverageResult(BaseModel):
    status: CoverageStatus
    total_cost: float
    covered_amount: float
    out_of_pocket: float
    steps: list[CalculationStep]
    missing_terms: list[str]
    assumptions: list[str]
    cost_estimate: CostEstimate | None = None
    disclaimer: str
