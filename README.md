# policy-to-patient
AI-powered Insurance Coverage and Treatment Cost Intelligence for HackMatrix 5.0 FIN-01

> **Prototype.** Outputs are estimates, not insurance decisions. Treatment costs come from a
> **synthetic** demo table (hand-written placeholder numbers, not real prices). No accuracy
> metrics are claimed.

## What works (Phase 1)

```
Policy PDF → page-level text → page/section-aware chunks → TF-IDF retrieval
→ evidence-backed Q&A with citations → synthetic cost estimate → deterministic coverage/OOP
```

| Layer | Module | Responsibility |
|---|---|---|
| Ingestion | `backend/app/ingestion/` | `pypdf` text per page (1-based), heading-aware chunks that never cross pages |
| Retrieval | `backend/app/retrieval/` | `Retriever` interface; TF-IDF implementation with scores |
| Q&A | `backend/app/qa/` | Sends **only** retrieved passages to Claude, validates `[C#]` citations |
| Cost | `backend/app/cost/` | `CostEstimator` interface; synthetic CSV lookup |
| Rules | `backend/app/rules/` | Deterministic deductible / co-pay / limits / exclusions / waiting periods |
| API | `backend/app/api/`, `main.py` | FastAPI |
| Storage | `backend/app/store.py` | JSON files under `data/runtime/` (git-ignored) |

See `docs/PHASE1_PLAN.md` for the design and acceptance criteria.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e "backend[dev]"
```

Optional LLM (without it, `/ask` returns the cited evidence passages and no generated answer):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_MODEL=claude-opus-5      # default
```

Other settings (all optional) are listed in `backend/.env.example`.

## Run the API

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

## Run the tests

```bash
cd backend
pytest -q
```

Tests run offline with no API key. They build a small **fictional** policy PDF in code
(`backend/tests/pdf_factory.py`), and use a stub LLM for the generative path.

## Demo flow (curl)

Use a real, publicly available policy wording PDF (none is committed to this repo).

```bash
B=http://localhost:8000

# 1. Upload + index a policy → returns doc_id, page_count, chunk_count, sections
curl -F "file=@policy.pdf;type=application/pdf" $B/policies

# 2. Inspect chunks on a page, or search evidence
curl "$B/policies/<doc_id>/chunks?page=4"
curl -X POST $B/policies/<doc_id>/search -H 'content-type: application/json' \
     -d '{"query": "room rent limit"}'

# 3. Ask a question → answer (if LLM configured) + citations with page/section
curl -X POST $B/policies/<doc_id>/ask -H 'content-type: application/json' \
     -d '{"question": "What is the waiting period for maternity?"}'

# 4. Treatment cost (SYNTHETIC)
curl $B/treatments
curl -X POST $B/estimate -H 'content-type: application/json' \
     -d '{"treatment_code": "APPENDECTOMY", "city_tier": "tier2", "room_type": "private"}'

# 5. Coverage / out-of-pocket (deterministic). Terms come from the policy; cite the chunk they came from.
curl -X POST $B/coverage -H 'content-type: application/json' -d '{
  "treatment": {"treatment_code": "APPENDECTOMY", "city_tier": "tier2", "room_type": "private"},
  "terms": {"sum_insured": 500000, "deductible": 10000, "copay_pct": 10,
            "room_rent_limit_per_day": 5000,
            "citations": {"deductible": {"chunk_id": "<chunk_id>", "page": 4}}}
}'
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness; `llm_mode` is `generative` or `extractive` |
| POST | `/policies` | Upload a text PDF; extract, chunk, store |
| GET | `/policies` | List ingested policies |
| GET | `/policies/{doc_id}` | Policy summary |
| GET | `/policies/{doc_id}/chunks?page=N` | Chunks with page/section/offsets |
| POST | `/policies/{doc_id}/search` | Top evidence passages with scores |
| POST | `/policies/{doc_id}/ask` | Grounded Q&A |
| GET | `/treatments` | Treatments in the synthetic cost table |
| POST | `/estimate` | Synthetic cost range for a treatment |
| POST | `/coverage` | Itemised coverage / OOP breakdown |

### `/ask` status values

| status | Meaning |
|---|---|
| `answered` | LLM answer; every cited `[C#]` maps to a retrieved passage |
| `evidence_only` | No API key: relevant passages returned, no generated answer |
| `insufficient_evidence` | No passage above `MIN_EVIDENCE_SCORE`, the LLM said the passages don't answer, or the answer cited nothing / cited a passage it wasn't given (answer discarded) |
| `llm_error` | LLM call failed or was declined; passages returned instead |

### Coverage rules (fixed order)

1. Exclusion → 2. Waiting period → 3. Room-rent cap (excess only) → 4. Category sub-limit →
5. Deductible → 6. Co-pay → 7. Coverage limit (remaining sum insured).

Missing `sum_insured`, `deductible` or `copay_pct` are listed in `missing_terms` (status
`incomplete_terms`) and never guessed. Money is computed with `Decimal`, rounded to 2 dp.

## Known limitations

- Policy terms for `/coverage` are entered by the user; automatic extraction from the PDF is not implemented.
- TF-IDF is lexical: synonyms are missed and generic words can match unrelated passages. The
  evidence threshold plus the LLM/citation check are the safeguards. Embeddings are planned for Phase 2.
- Scanned PDFs: pages without a text layer are reported in `empty_pages`; there is no OCR.
- Heading detection is heuristic; `section` is `null` when no heading is found.
- Cost figures are synthetic. There is no trained model and no evaluation metric yet.
- Not implemented: bill auditing, self-healing retrieval loop, authentication, frontend.
