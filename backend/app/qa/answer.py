"""Grounded Q&A: retrieve evidence → (optionally) explain it with an LLM → validate citations."""

from __future__ import annotations

import re

from app.models import AskResponse, Citation, Evidence
from app.qa.llm import LLMClient, LLMError
from app.qa.prompt import INSUFFICIENT_MARKER, SYSTEM_PROMPT, build_user_prompt, passage_label
from app.retrieval.base import Retriever, ScoredChunk

QUOTE_CHARS = 300
_CITATION_GROUP = re.compile(r"\[\s*(C\d+(?:\s*[,;]\s*C\d+)*)\s*\]")


def extract_citation_labels(text: str) -> list[str]:
    """Labels cited in the answer, in first-appearance order (e.g. ['C2', 'C1'])."""
    labels: list[str] = []
    for group in _CITATION_GROUP.findall(text):
        for label in re.split(r"\s*[,;]\s*", group):
            if label not in labels:
                labels.append(label)
    return labels


def validate_citations(answer: str, evidence: list[ScoredChunk]) -> tuple[list[str], list[str]]:
    """Split cited labels into (valid, invalid) against the passages actually provided."""
    allowed = {passage_label(i) for i in range(len(evidence))}
    cited = extract_citation_labels(answer)
    return [c for c in cited if c in allowed], [c for c in cited if c not in allowed]


def _to_evidence(hit: ScoredChunk) -> Evidence:
    c = hit.chunk
    return Evidence(
        chunk_id=c.chunk_id, doc_name=c.doc_name, page=c.page, section=c.section,
        text=c.text, score=hit.score,
    )


def _to_citation(label: str, hit: ScoredChunk) -> Citation:
    c = hit.chunk
    quote = c.text if len(c.text) <= QUOTE_CHARS else c.text[:QUOTE_CHARS].rstrip() + "…"
    return Citation(
        ref=label, chunk_id=c.chunk_id, doc_name=c.doc_name, page=c.page,
        section=c.section, quote=quote,
    )


def answer_question(
    question: str,
    doc_id: str,
    retriever: Retriever,
    llm: LLMClient | None,
    top_k: int,
    min_score: float,
) -> AskResponse:
    mode = "generative" if llm is not None else "extractive"
    hits = retriever.search(doc_id, question, top_k)
    evidence = [h for h in hits if h.score >= min_score]

    if not evidence:
        best = f" (best score {hits[0].score:.3f})" if hits else ""
        return AskResponse(
            status="insufficient_evidence", mode=mode, question=question,
            evidence=[_to_evidence(h) for h in hits],
            message=(
                f"No passage met the minimum relevance score {min_score}{best}. "
                "The retrieved policy text does not clearly address this question; "
                "check the policy document manually."
            ),
        )

    evidence_out = [_to_evidence(h) for h in evidence]
    all_citations = [_to_citation(passage_label(i), h) for i, h in enumerate(evidence)]

    if llm is None:
        return AskResponse(
            status="evidence_only", mode=mode, question=question,
            citations=all_citations, evidence=evidence_out,
            message=(
                "No LLM is configured, so no answer was generated. "
                "These are the most relevant policy passages; read them to answer the question."
            ),
        )

    try:
        reply = llm.complete(SYSTEM_PROMPT, build_user_prompt(question, evidence))
    except LLMError as exc:
        return AskResponse(
            status="llm_error", mode=mode, question=question,
            citations=all_citations, evidence=evidence_out,
            message=f"{exc}. Showing the retrieved policy passages instead.",
        )

    if INSUFFICIENT_MARKER in reply:
        return AskResponse(
            status="insufficient_evidence", mode=mode, question=question,
            evidence=evidence_out,
            message="The retrieved passages do not answer this question according to the LLM.",
        )

    valid, invalid = validate_citations(reply, evidence)
    if invalid or not valid:
        problem = (
            f"cited passages that were not provided ({', '.join(invalid)})"
            if invalid
            else "did not cite any passage"
        )
        return AskResponse(
            status="insufficient_evidence", mode=mode, question=question,
            evidence=evidence_out,
            message=f"The generated answer {problem}, so it was discarded.",
        )

    by_label = {passage_label(i): h for i, h in enumerate(evidence)}
    return AskResponse(
        status="answered", mode=mode, question=question, answer=reply,
        citations=[_to_citation(label, by_label[label]) for label in valid],
        evidence=evidence_out,
        message="Answer generated only from the cited policy passages. Verify against the source pages.",
    )
