"""Evaluation dataset: a fictional policy wording plus labelled questions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.config import REPO_ROOT
from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_extract import extract_pages
from app.models import Chunk
from evaluation.synthetic_pdf import make_text_pdf

DEFAULT_DATASET = REPO_ROOT / "data" / "eval" / "fictional_policy_eval.json"
EVAL_DOC_ID = "evalpolicy"


class Location(BaseModel):
    page: int
    section: str


class Question(BaseModel):
    id: str
    kind: Literal["direct", "paraphrase", "unanswerable"]
    question: str
    answerable: bool
    expected: list[Location] = Field(default_factory=list)
    keyword: str | None = None
    absent_terms: list[str] = Field(default_factory=list)


class PolicyText(BaseModel):
    doc_name: str
    pages: list[list[str]]


class Dataset(BaseModel):
    name: str
    description: str
    policy: PolicyText
    questions: list[Question]

    def policy_pdf(self) -> bytes:
        return make_text_pdf(self.policy.pages)

    def chunks(self) -> list[Chunk]:
        """Run the fictional policy through the real ingestion path (PDF → pages → chunks)."""
        pages = extract_pages(self.policy_pdf())
        return chunk_pages(pages, doc_id=EVAL_DOC_ID, doc_name=self.policy.doc_name)


def load_dataset(path: Path = DEFAULT_DATASET) -> Dataset:
    return Dataset.model_validate(json.loads(Path(path).read_text()))


def is_hit(chunk: Chunk, question: Question) -> bool:
    return any(chunk.page == loc.page and chunk.section == loc.section for loc in question.expected)


def validate_labels(dataset: Dataset, chunks: list[Chunk]) -> list[str]:
    """Check every label against the ingested text. Returns a list of problems (empty = valid)."""
    problems: list[str] = []
    ids = [q.id for q in dataset.questions]
    if len(ids) != len(set(ids)):
        problems.append("duplicate question ids")
    sections = {c.section for c in chunks}
    full_text = "\n".join(c.text for c in chunks).lower()
    for q in dataset.questions:
        if q.answerable:
            if not q.expected or not q.keyword:
                problems.append(f"{q.id}: answerable question needs 'expected' and 'keyword'")
                continue
            for loc in q.expected:
                if loc.section not in sections:
                    problems.append(f"{q.id}: section {loc.section!r} not produced by the chunker")
            labelled = [c for c in chunks if is_hit(c, q)]
            if not any(q.keyword.lower() in c.text.lower() for c in labelled):
                problems.append(f"{q.id}: keyword {q.keyword!r} not found in the labelled chunk(s)")
        else:
            if q.expected or not q.absent_terms:
                problems.append(f"{q.id}: unanswerable question needs 'absent_terms' and no 'expected'")
            for term in q.absent_terms:
                if term.lower() in full_text:
                    problems.append(f"{q.id}: 'unanswerable' but {term!r} appears in the policy")
    return problems
