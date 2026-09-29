"""Prompt construction for grounded policy Q&A."""

from __future__ import annotations

from app.qa.verify import passage_label
from app.retrieval.base import ScoredChunk

INSUFFICIENT_MARKER = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT = f"""You explain health-insurance policy wording to a patient.

You will receive numbered passages retrieved from ONE policy document, then a question.
Rules:
- Use only the passages. Do not use outside knowledge about insurance or any insurer.
- Cite the passage label in square brackets, e.g. [C2], right after every statement it supports.
- Cite only labels that appear in the passages.
- Quote amounts, percentages and time periods exactly as written.
- Do not calculate payouts or out-of-pocket amounts; a separate calculator does that.
- If the passages do not answer the question, reply with exactly {INSUFFICIENT_MARKER} and nothing else.
- Keep the answer short: at most a few sentences."""


def build_user_prompt(question: str, evidence: list[ScoredChunk]) -> str:
    blocks = []
    for i, hit in enumerate(evidence):
        c = hit.chunk
        header = f"[{passage_label(i)}] page {c.page}" + (f", section: {c.section}" if c.section else "")
        blocks.append(f"{header}\n{c.text}")
    passages = "\n\n".join(blocks)
    return f"<passages>\n{passages}\n</passages>\n\nQuestion: {question}"
