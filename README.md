# policy-to-patient
AI-powered Insurance Coverage and Treatment Cost Intelligence for HackMatrix 5.0 FIN-01

> **Prototype.** Outputs are estimates, not insurance decisions. Treatment costs come from a
> **synthetic** demo table (hand-written placeholder numbers, not real prices). Retrieval has
> only been measured on a **fictional** policy (see `docs/PHASE3_EVALUATION.md`); no real-world
> accuracy is claimed.

## What works (Phases 1–3)

```
Policy PDF → page-level text → page/section-aware chunks → embeddings (once, at upload)
→ persistent vector store → per-policy search → evidence above a relevance threshold
   └─ weak evidence? Claude rewrites the query (bounded) → search again
→ Claude answers from that evidence only → deterministic verification of citations and figures
Treatment input → synthetic cost estimate → deterministic coverage/OOP
Minimal web UI for the whole flow; retrieval evaluation on a labelled fictional policy
```

| Layer | Module | Responsibility |
|---|---|---|
| Ingestion | `backend/app/ingestion/` | `pypdf` text per page (1-based), heading-aware chunks that never cross pages |
| Embeddings | `backend/app/embeddings/` | `Embedder` interface; `fastembed` (local `bge-small-en-v1.5`) or `hashing` (offline, lexical) |
| Vector store | `backend/app/vectorstore/` | `VectorStore` interface; PostgreSQL + pgvector, or local `.npz` files |
| Retrieval | `backend/app/retrieval/` | `Retriever` interface: `index(doc_id, chunks)` once, `search(doc_id, query, k)` |
| Q&A | `backend/app/qa/` | Self-correcting retrieval (`answer.py`, `rewrite.py`); only retrieved passages go to Claude; answers verified (`verify.py`) |
| Cost | `backend/app/cost/` | `CostEstimator` interface; synthetic CSV lookup |
| Rules | `backend/app/rules/` | Deterministic deductible / co-pay / limits / exclusions / waiting periods |
| API | `backend/app/api/`, `main.py` | FastAPI |
| Storage | `backend/app/store.py` | Uploaded PDF, pages and chunks as JSON under `data/runtime/` (git-ignored) |
| Evaluation | `backend/evaluation/`, `data/eval/` | Labelled fictional policy, retrieval metrics, CLI |
| UI | `frontend/` | React + TypeScript (Vite), served by the API at `/ui/` |

Design notes: `docs/PHASE1_PLAN.md`, `docs/PHASE2_NOTES.md`, `docs/PHASE3_EVALUATION.md`.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e "backend[embeddings,pgvector,dev]"

# UI (Node 18+)
cd frontend && npm install && npm run build && cd ..
```

Extras: `embeddings` = local semantic model via fastembed (downloads ~70 MB from Hugging Face on
first start into `data/runtime/models/`); `pgvector` = PostgreSQL driver; `dev` = test tools;
`e2e` = Playwright for the browser test.

### Embeddings

| `EMBEDDING_PROVIDER` | What it is | When to use |
|---|---|---|
| `fastembed` (default) | Local semantic model `BAAI/bge-small-en-v1.5`, CPU, no API key | Normal use |
| `hashing` | Deterministic feature hashing. **Lexical, not semantic** | Tests, or offline when the model can't be downloaded |

Each policy is embedded **once** at upload. Vectors are stored with the embedding model's name,
so restarting the API never re-embeds. Changing `EMBEDDING_PROVIDER`/`EMBEDDING_MODEL` makes each
policy re-index once from its stored chunks on first use.

### Vector store: PostgreSQL + pgvector (recommended)

Set `DATABASE_URL` to use pgvector; without it, vectors are stored as local files under
`DATA_DIR/vectors/`. Tables are created automatically on startup. For example, with Docker:

```bash
docker run -d --name p2p-db -e POSTGRES_USER=p2p -e POSTGRES_PASSWORD=p2p -e POSTGRES_DB=p2p \
  -p 5432:5432 pgvector/pgvector:pg16
export DATABASE_URL=postgresql://p2p:p2p@localhost:5432/p2p
```

Or on Debian/Ubuntu with PostgreSQL 16: `apt install postgresql-16-pgvector`, then create a
database and user (the app runs `CREATE EXTENSION IF NOT EXISTS vector`, which needs a role
allowed to create extensions).

Optional LLM (without it, `/ask` returns the cited evidence passages and no generated answer,
and the query-rewrite retry is skipped):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_MODEL=claude-opus-5      # default
```

Or use Gemini instead (`pip install -e "backend[gemini]"`):

```bash
export LLM_PROVIDER=gemini          # default: anthropic
export GEMINI_API_KEY=...
export GEMINI_MODEL=gemini-2.5-flash   # default
```

Both providers receive the same prompt and evidence and go through the same citation and figure
verification.

Other settings (all optional) are listed in `backend/.env.example`.

## Run the API

```bash
cd backend
uvicorn app.main:create_app --factory --reload --port 8000
```

- UI: http://localhost:8000/ui/ (after `npm run build`; `/` redirects there)
- API docs: http://localhost:8000/docs
- UI development with hot reload: `cd frontend && npm run dev` → http://localhost:5173/ui/
  (forwards API calls to `http://localhost:8000`; override with `API_URL`)

## Demo (5 minutes)

```bash
cd backend
python -m evaluation.demo_pdf ../data/runtime/fictional_health_shield_policy.pdf   # FICTIONAL policy
uvicorn app.main:create_app --factory --port 8000     # add EMBEDDING_PROVIDER=hashing if offline
```

Open http://localhost:8000/ui/ and:

1. Upload `data/runtime/fictional_health_shield_policy.pdf` → 8 pages, 16 chunks indexed.
2. Ask "What is the deductible for each policy year?" → page 4, section "8. DEDUCTIBLE" cited.
3. Ask "Is there cover for vaccinations?" → **Insufficient evidence** (the policy doesn't say).
4. Treatment "Cataract surgery", Estimate cost → ₹40,000 **synthetic** estimate.
5. Coverage: sum insured 500000, deductible 10000, co-pay 10, cataract sub-limit 40000 →
   insurer ₹27,000, you ₹13,000, with each rule shown.
6. Stop and restart the API: the policy is still listed and is not re-embedded.

With `ANTHROPIC_API_KEY` set, step 2 returns a Claude-written answer; a figure Claude states
that is not in the cited passage sends the answer to **manual review** instead.

## Run the tests

```bash
cd backend
pytest -q
```

Tests run offline with no API key or model download. They build a small **fictional** policy
PDF in code (`backend/tests/pdf_factory.py`), use a stub LLM for the generative path, the
`hashing` embedder, and a tiny deterministic "concept" embedder (in `tests/conftest.py`) that
stands in for a semantic model. That stand-in checks the plumbing; it says nothing about any
real model's retrieval quality.

The vector-store contract tests also run against PostgreSQL + pgvector when a test database is
available (its tables are truncated):

```bash
TEST_DATABASE_URL=postgresql://p2p:p2p@localhost:5432/p2p_test pytest -q
```

Browser test of the UI against a running server (see `backend/tests/e2e/test_ui_e2e.py`):

```bash
pip install -e "backend[e2e]" && playwright install chromium
E2E_BASE_URL=http://localhost:8000 pytest tests/e2e -q
```

## Retrieval evaluation

```bash
cd backend
python -m evaluation.run --embedder hashing              # offline lexical baseline
python -m evaluation.run --embedder fastembed --json out.json   # real semantic model
```

48 hand-labelled questions (21 direct, 16 paraphrased, 11 unanswerable) against a **fictional**
8-page policy (`data/eval/fictional_policy_eval.json`). Labels are checked against the ingested
text by a test. Reports hit@1/3/5, MRR, answer-gate behaviour and a threshold sweep. Methodology
and current results: `docs/PHASE3_EVALUATION.md`.

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
| GET | `/health` | Liveness; `llm_mode`, active `retrieval` (embedder @ store), `min_evidence_score` |
| POST | `/policies` | Upload a text PDF; extract, chunk, embed, store |
| GET | `/policies` | List ingested policies |
| GET | `/policies/{doc_id}` | Policy summary |
| GET | `/policies/{doc_id}/chunks?page=N` | Chunks with page/section/offsets |
| POST | `/policies/{doc_id}/search` | Top passages of this policy only, with cosine scores |
| POST | `/policies/{doc_id}/ask` | Grounded Q&A |
| GET | `/treatments` | Treatments in the synthetic cost table |
| POST | `/estimate` | Synthetic cost range for a treatment |
| POST | `/coverage` | Itemised coverage / OOP breakdown |

### `/ask` status values

| status | Meaning |
|---|---|
| `answered` | Claude's answer passed verification: every citation is a supplied passage and every figure appears in its cited passage |
| `evidence_only` | No API key: relevant passages returned, no generated answer |
| `insufficient_evidence` | No passage above `MIN_EVIDENCE_SCORE` (even after query rewrites), or Claude said the passages don't answer |
| `manual_review` | Evidence was found but Claude's answer failed verification; the answer is withheld and `verification_issues` explains why |
| `llm_error` | LLM call failed or was declined; passages returned instead |

Every response includes `retrieval_attempts`: the original query and any rewritten queries, each
with its best score and whether it found evidence. Rewrites happen only when an LLM is configured,
at most `MAX_QUERY_REWRITES` times (default 1, hard cap 2). The rewritten query is used only for
searching; Claude always answers the user's original question from stored policy text.

### Coverage rules (fixed order)

1. Exclusion → 2. Waiting period → 3. Room-rent cap (excess only) → 4. Category sub-limit →
5. Deductible → 6. Co-pay → 7. Coverage limit (remaining sum insured).

Missing `sum_insured`, `deductible` or `copay_pct` are listed in `missing_terms` (status
`incomplete_terms`) and never guessed. Money is computed with `Decimal`, rounded to 2 dp.

## Known limitations

- Policy terms for `/coverage` are entered by the user; automatic extraction from the PDF is not implemented.
- **The semantic model has not been evaluated yet.** The fastembed model could not be downloaded
  in the development environment, so its `0.55` threshold is uncalibrated. Only the `hashing`
  baseline has measured results, on a fictional policy (`docs/PHASE3_EVALUATION.md`).
- The `hashing` embedder is lexical: on the evaluation set it finds the right passage for every
  directly worded question but only 75% of paraphrases within the top 5, and lets about half of
  the unanswerable questions through the gate (they then show passages, never an answer).
- Answer verification is deterministic: it catches wrong/missing citations and figures not in
  the cited text, but not a wrong claim without figures (that relies on Claude replying
  `INSUFFICIENT_EVIDENCE`). The live Claude path has only been tested with mocks.
- Retrieval is vector-only (no hybrid keyword + vector ranking).
- Scanned PDFs: pages without a text layer are reported in `empty_pages`; there is no OCR.
- Heading detection is heuristic; `section` is `null` when no heading is found.
- Cost figures are synthetic. There is no trained model and no evaluation metric yet.
- Not implemented: bill auditing, OCR, automatic extraction of policy terms, cost ML model,
  authentication, deployment.
