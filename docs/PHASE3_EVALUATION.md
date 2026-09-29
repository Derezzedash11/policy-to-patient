# Phase 3 — Retrieval Evaluation, Self-Correcting Retrieval, Demo

## What Phase 3 added

| Area | Where | Summary |
|---|---|---|
| Evaluation set | `data/eval/fictional_policy_eval.json` | Fictional 8-page policy + 48 labelled questions |
| Evaluation code | `backend/evaluation/` | Label validation, metrics, threshold sweep, CLI (`python -m evaluation.run`) |
| Self-correcting retrieval | `backend/app/qa/answer.py`, `rewrite.py` | Weak evidence → Claude rewrites the query (≤ 2 times) → search again |
| Answer verification | `backend/app/qa/verify.py` | Citations must be supplied passages; figures must appear in the cited text; else `manual_review` |
| UI | `frontend/` | React + TypeScript (Vite); upload → ask → citations → estimate → coverage |
| Browser test | `backend/tests/e2e/` | Playwright run of the full demo workflow against a live server |

## Evaluation methodology

**Data.** No public policy wording could be downloaded in the development environment (insurer
and regulator sites are blocked by its network policy), so the set uses a **fictional** policy
written for testing. It follows a typical Indian health-policy structure: sum insured, room rent,
sub-limits, pre/post hospitalisation, deductible, co-payment, waiting periods, maternity,
exclusions, claims, renewal, grievances. It does not describe any real product. The policy goes
through the real pipeline (PDF → page extraction → chunker), giving 16 chunks.

**Questions (48).**

| Kind | n | Purpose |
|---|---|---|
| direct | 21 | Wording overlaps with the policy text |
| paraphrase | 16 | Patient phrasing with little or no word overlap (e.g. "tooth extraction" → dental exclusion) |
| unanswerable | 11 | Topics the policy does not mention (vaccination, AYUSH, organ donor, …) |

Each answerable question lists the acceptable page + section and a keyword; each unanswerable
question lists terms that must not occur in the policy. `validate_labels` checks every label
against the ingested chunks (keyword inside the labelled chunk; absent terms really absent), and a
test fails if any label is wrong. It caught one labelling mistake while the set was being built.

**Metrics.**

- **hit@k**: a chunk with a labelled page *and* section is among the top k results (k = 1, 3, 5).
- **keyword hit@5**: a top-5 chunk contains the labelled keyword.
- **MRR@5**: mean reciprocal rank of the first labelled chunk.
- **Answer gate**: the pipeline's status per question: do answerable questions pass the relevance
  threshold, is the labelled passage among the evidence that would be sent to Claude, and are
  unanswerable questions rejected as `insufficient_evidence`?
- **Threshold sweep**: top-1 score threshold vs. answerable-pass / unanswerable-reject rates.

**Caveats.** One fictional document and questions written by the same author as the policy.
These numbers compare configurations and catch regressions. They are **not** an estimate of
accuracy on real policies.

## Results

### Real embedding model: not executed

`python -m evaluation.run --embedder fastembed` was attempted and failed: the model download from
Hugging Face is blocked by the environment's network policy (`403 Forbidden` from the egress
proxy). **No semantic-model results exist yet**, and the `0.55` default threshold for
`bge-small-en-v1.5` remains uncalibrated. Run the command above on a machine that can reach
huggingface.co to produce them.

### Lexical baseline (`hashing` embedder), measured

`python -m evaluation.run --embedder hashing` (top_k 5, min_score 0.10, no LLM, so no rewrites).
Identical results with the local-file store and PostgreSQL + pgvector.

| Answerable subset | n | hit@1 | hit@3 | hit@5 | keyword hit@5 | MRR@5 |
|---|---|---|---|---|---|---|
| all answerable | 37 | 0.70 | 0.81 | 0.89 | 0.89 | 0.77 |
| direct | 21 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| paraphrase | 16 | 0.31 | 0.56 | 0.75 | 0.75 | 0.46 |

Answer gate at 0.10:

- answerable questions passing the gate: 0.81
- answerable questions whose labelled passage is among the evidence sent: 0.70
- unanswerable questions rejected: 0.46 (the other 6 show weakly matching passages as evidence;
  with no LLM they never produce an answer, and with an LLM the answer still has to pass
  `INSUFFICIENT_EVIDENCE` handling and verification)

Top-5 misses: p02 "What share of each bill do I have to pay myself?", p03 "hospital bed per night",
p05 "tooth extraction", p15 "senior citizens pay a larger share". All are paraphrases with no
shared vocabulary: the expected failure mode of a lexical method.

Threshold sweep (top-1 score):

| threshold | answerable pass | unanswerable reject | balanced accuracy |
|---|---|---|---|
| 0.10 (default) | 0.81 | 0.45 | 0.63 |
| 0.20 | 0.65 | 0.55 | 0.60 |
| 0.25 | 0.54 | 0.91 | 0.73 |
| 0.30 | 0.54 | 1.00 | 0.77 |

A two-fold split (choose on half the questions, test on the other half) picked 0.30 and 0.25,
with held-out balanced accuracy 0.78 and 0.68. **Decision: keep 0.10 for the hashing embedder.**
Its score distributions overlap too much for any threshold to separate answerable from
unanswerable questions well. 0.30 would reject about half of the answerable questions (46%),
including directly worded ones. At 0.10, unrelated passages are shown as evidence rather than
answers going missing, and Claude + verification form the second gate. Revisit this with the
semantic model, which is what this set is meant to calibrate.

### Self-correcting retrieval

Not measured on the evaluation set: it needs an LLM to rewrite queries, and no API key is
available here. Its control flow is covered by tests with scripted LLMs: recovery via a
rewritten query, rewrite that still finds nothing, bounded retries, repeated/empty rewrites,
rewrite errors, disabled when `MAX_QUERY_REWRITES=0`.

One behaviour found while testing: with the lexical embedder, the rewrite "vaccination cover
limit" weakly matched the maternity section. A fabricated amount in the mock answer was then
caught by the figure check (`manual_review`). A fabricated claim **without** figures would not be
caught deterministically. That case relies on Claude following the instruction to reply
`INSUFFICIENT_EVIDENCE`, which could not be tested live here.

## End-to-end verification performed

- Test suite: 161 tests pass with `TEST_DATABASE_URL` and `E2E_BASE_URL` set (pgvector + browser);
  without them, 151 pass and 10 skip (9 pgvector contract/API tests, 1 browser test).
- Live server (PostgreSQL + pgvector, hashing embedder, no API key), exercised with curl:
  - two policies uploaded, each question's citations come only from its own policy;
  - error paths: non-PDF / corrupt PDF → 422, unknown policy → 404, short question → 422,
    unknown treatment → 404, conflicting coverage inputs → 422, path-traversal id → 404;
  - restart: `policy_index.indexed_at` unchanged, identical search results, 0 re-index events,
    both policies still listed.
- Browser (Playwright, Chromium) against the same server, before and after the restart: upload →
  "8 pages, 16 chunks" → deductible question cites page 4 "8. DEDUCTIBLE" → vaccination question
  shows "Insufficient evidence" → cataract synthetic estimate ₹40,000 → coverage ₹27,000 / ₹13,000
  with rules sub_limit → deductible → copay → coverage_limit.

## Real Gemini API validation

After the Gemini provider was added (`LLM_PROVIDER=gemini`), the full pipeline was run against
the **real Gemini API**. This is separate from the unit tests, which use a fake client
(`tests/test_gemini_llm.py`, 23 tests) and never call the API.

Setup: model `gemini-2.5-flash`; API server started with `uvicorn app.main:create_app --factory`;
`EMBEDDING_PROVIDER=hashing` (keyword retrieval; see below); local-file vector store; the
fictional demo policy (8 pages, 16 chunks); questions sent to `POST /policies/{id}/ask`.

| Test | Question | Observed result |
|---|---|---|
| Direct | "What is the deductible for each policy year?" | `answered`: "A deductible of Rs 10,000 applies to each policy year [C1]…", cited page 4, "8. DEDUCTIBLE"; citation and figure checks passed |
| Unsupported | "Is there cover for vaccinations?" | `insufficient_evidence`: no passage above the threshold; Gemini's rewritten query "immunization coverage" also found none; no answer, no citations |
| Unsupported | "Are organ donor expenses covered?" | `insufficient_evidence`: keyword retrieval let an unrelated passage through (score 0.29), and Gemini replied `INSUFFICIENT_EVIDENCE` instead of inventing cover |
| Weak evidence | "What share of each bill do I have to pay myself?" | `answered` after retry: first search scored 0.044; Gemini rewrote the query to "co-payment" (0.706); answer cited "9. CO-PAYMENT" and "8. DEDUCTIBLE", with every figure found in its cited passage |
| Paraphrase | "How much of my own money do I spend each year before the insurance starts paying?" | `insufficient_evidence`: keyword retrieval surfaced only "3. SUM INSURED", not the deductible passage, and Gemini correctly declined. The retry only runs when no passage clears the threshold, so it did not run here |

- The free-tier quota (5 requests per minute for `gemini-2.5-flash`) was hit once with HTTP 429
  during the weak-evidence test. The app returned `llm_error` with the evidence passages, and the
  test passed when re-run after the quota window reset.
- This is one run on one fictional document, not an accuracy measurement.

## Not verified

- The fastembed semantic model: the Hugging Face model download was unavailable in the
  development environment, so no semantic retrieval results exist.
- The real Claude API (no key was available). The `AnthropicLLM` wrapper is unit-tested with a
  fake SDK client; generation, rewriting and verification are tested with scripted mocks.
- Any real insurance policy.
