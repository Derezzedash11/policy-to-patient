"""Query rewriting for the self-correcting retrieval step.

The LLM only produces a search query. Its output is never shown as an answer or used as
evidence: evidence always comes from the stored policy chunks.
"""

from __future__ import annotations

from app.qa.llm import LLMClient

REWRITE_SYSTEM_PROMPT = """You turn a patient's question about a health insurance policy into a \
search query for the policy document.
Rules:
- Use the formal vocabulary policy wordings typically use (e.g. "co-payment", "room rent", \
"waiting period", "pre-existing disease", "exclusions", "sum insured").
- Output only the query, on one line, at most 20 words.
- Do not answer the question and do not add amounts, numbers or facts."""

MAX_QUERY_CHARS = 200


def rewrite_query(llm: LLMClient, question: str, tried: list[str]) -> str | None:
    """Return a new search query, or None if the model produced nothing usable."""
    tried_text = "\n".join(f"- {q}" for q in tried)
    reply = llm.complete(
        REWRITE_SYSTEM_PROMPT,
        f"Question: {question}\n\nQueries already tried (found no relevant passage):\n{tried_text}",
    )
    line = next((ln.strip() for ln in reply.splitlines() if ln.strip()), "")
    for prefix in ("query:", "search query:"):
        if line.lower().startswith(prefix):
            line = line[len(prefix):].strip()
    line = line.strip("\"'` ")[:MAX_QUERY_CHARS]
    if not line or line.lower() in {q.lower() for q in tried}:
        return None
    return line
