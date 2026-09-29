"""Grounded Q&A with a small self-correcting retrieval loop.

1. Retrieve passages for the question; keep those at or above the relevance threshold.
2. If none qualify and an LLM is configured: rewrite the query (at most `max_rewrites` times),
   retrieve again, and merge the results. The LLM only writes search queries here.
3. No qualifying passage → insufficient_evidence. No LLM → evidence_only (passages, no answer).
4. The LLM answers from the qualifying passages only; the answer is verified deterministically
   (citations must point at supplied passages, figures must appear in the cited text).
   Verification failure → manual_review, with the evidence and the reasons.
"""

from __future__ import annotations

from app.models import AskResponse, Citation, Evidence, RetrievalAttempt
from app.qa.llm import LLMClient, LLMError
from app.qa.prompt import INSUFFICIENT_MARKER, SYSTEM_PROMPT, build_user_prompt
from app.qa.rewrite import rewrite_query
from app.qa.verify import extract_citation_labels, passage_label, validate_citations, verify_answer
from app.retrieval.base import Retriever, ScoredChunk

__all__ = ["answer_question", "extract_citation_labels", "validate_citations"]

QUOTE_CHARS = 300
MAX_REWRITES_LIMIT = 2


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


def _merge(a: list[ScoredChunk], b: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
    """Union of two result lists, keeping each chunk's best score."""
    best: dict[str, ScoredChunk] = {}
    for hit in a + b:
        current = best.get(hit.chunk.chunk_id)
        if current is None or hit.score > current.score:
            best[hit.chunk.chunk_id] = hit
    return sorted(best.values(), key=lambda h: h.score, reverse=True)[:top_k]


def _retrieve_with_correction(
    question: str, doc_id: str, retriever: Retriever, llm: LLMClient | None,
    top_k: int, min_score: float, max_rewrites: int,
) -> tuple[list[ScoredChunk], list[RetrievalAttempt]]:
    hits = retriever.search(doc_id, question, top_k)
    attempts = [RetrievalAttempt(
        query=question, strategy="original", best_score=hits[0].score if hits else None,
        evidence_found=any(h.score >= min_score for h in hits),
    )]
    tried = [question]
    for _ in range(max(0, min(max_rewrites, MAX_REWRITES_LIMIT))):
        if attempts[-1].evidence_found or llm is None:
            break
        try:
            query = rewrite_query(llm, question, tried)
        except LLMError as exc:
            attempts.append(RetrievalAttempt(query="", strategy="llm_rewrite", note=f"rewrite failed: {exc}"))
            break
        if query is None:
            attempts.append(RetrievalAttempt(query="", strategy="llm_rewrite", note="no new query produced"))
            break
        tried.append(query)
        new_hits = retriever.search(doc_id, query, top_k)
        hits = _merge(hits, new_hits, top_k)
        attempts.append(RetrievalAttempt(
            query=query, strategy="llm_rewrite", best_score=new_hits[0].score if new_hits else None,
            evidence_found=any(h.score >= min_score for h in new_hits),
        ))
    return hits, attempts


def answer_question(
    question: str,
    doc_id: str,
    retriever: Retriever,
    llm: LLMClient | None,
    top_k: int,
    min_score: float,
    max_rewrites: int = 1,
) -> AskResponse:
    mode = "generative" if llm is not None else "extractive"
    hits, attempts = _retrieve_with_correction(
        question, doc_id, retriever, llm, top_k, min_score, max_rewrites
    )
    evidence = [h for h in hits if h.score >= min_score]
    base = dict(mode=mode, question=question, retrieval_attempts=attempts)

    if not evidence:
        best = f" (best score {hits[0].score:.3f})" if hits else ""
        retried = f" after {len(attempts) - 1} rewritten quer{'y' if len(attempts) == 2 else 'ies'}" \
            if len(attempts) > 1 else ""
        return AskResponse(
            status="insufficient_evidence", evidence=[_to_evidence(h) for h in hits], **base,
            message=(
                f"No passage met the minimum relevance score {min_score}{best}{retried}. "
                "The retrieved policy text does not clearly address this question; "
                "check the policy document manually."
            ),
        )

    evidence_out = [_to_evidence(h) for h in evidence]
    all_citations = [_to_citation(passage_label(i), h) for i, h in enumerate(evidence)]

    if llm is None:
        return AskResponse(
            status="evidence_only", citations=all_citations, evidence=evidence_out, **base,
            message=(
                "No LLM is configured, so no answer was generated. "
                "These are the most relevant policy passages; read them to answer the question."
            ),
        )

    try:
        reply = llm.complete(SYSTEM_PROMPT, build_user_prompt(question, evidence))
    except LLMError as exc:
        return AskResponse(
            status="llm_error", citations=all_citations, evidence=evidence_out, **base,
            message=f"{exc}. Showing the retrieved policy passages instead.",
        )

    if INSUFFICIENT_MARKER in reply:
        return AskResponse(
            status="insufficient_evidence", evidence=evidence_out, **base,
            message="The retrieved passages do not answer this question according to the LLM.",
        )

    valid, problems = verify_answer(reply, evidence)
    if problems:
        return AskResponse(
            status="manual_review", citations=all_citations, evidence=evidence_out,
            verification_issues=problems, **base,
            message=(
                "A generated answer failed verification against the policy text and was withheld. "
                "Review the passages below manually."
            ),
        )

    by_label = {passage_label(i): h for i, h in enumerate(evidence)}
    return AskResponse(
        status="answered", answer=reply,
        citations=[_to_citation(label, by_label[label]) for label in valid],
        evidence=evidence_out, **base,
        message="Answer generated only from the cited policy passages. Verify against the source pages.",
    )
