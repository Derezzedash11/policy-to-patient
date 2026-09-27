# policy-to-patient
AI-powered Insurance Coverage and Treatment Cost Intelligence for HackMatrix 5.0 FIN-01

> **Prototype.** Outputs are estimates, not insurance decisions. Treatment costs come from a
> **synthetic** demo table (hand-written placeholder numbers, not real prices). No accuracy
> metrics are claimed.

## What works (Phases 1–2)

```
Policy PDF → page-level text → page/section-aware chunks → embeddings (once, at upload)
→ persistent vector store → per-policy semantic search → evidence above a relevance threshold
→ grounded Claude answer with verified [C#] citations
Treatment input → synthetic cost estimate → deterministic coverage/OOP
```

| Layer | Module | Responsibility |
|---|---|---|
| Ingestion | `backend/app/ingestion/` | `pypdf` text per page (1-based), heading-aware chunks that never cross pages |
| Embeddings | `backend/app/embeddings/` | `Embedder` interface; `fastembed` (local `bge-small-en-v1.5`) or `hashing` (offline, lexical) |
| Vector store | `backend/app/vectorstore/` | `VectorStore` interface; PostgreSQL + pgvector, or local `.npz` files |
| Retrieval | `backend/app/retrieval/` | `Retriever` interface: `index(doc_id, chunks)` once, `search(doc_id, query, k)` |
| Q&A | `backend/app/qa/` | Sends **only** retrieved passages to Claude, validates `[C#]` citations |
| Cost | `backend/app/cost/` | `CostEstimator` interface; synthetic CSV lookup |
| Rules | `backend/app/rules/` | Deterministic deductible / co-pay / limits / exclusions / waiting periods |
| API | `backend/app/api/`, `main.py` | FastAPI |
| Storage | `backend/app/store.py` | Uploaded PDF, pages and chunks as JSON under `data/runtime/` (git-ignored) |

See `docs/PHASE1_PLAN.md` and `docs/PHASE2_NOTES.md` for design decisions.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e "backend[embeddings,pgvector,dev]"
```

Extras: `embeddings` = local semantic model via fastembed (downloads ~70 MB from Hugging Face on
first start into `data/runtime/models/`); `pgvector` = PostgreSQL driver; `dev` = test tools.

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

Optional LLM (without it, `/ask` returns the cited evidence passages and no generated answer):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_MODEL=claude-opus-5      # default
```

Other settings (all optional) are listed in `backend/.env.example`.

## Run the API

```bash
cd backend
uvicorn app.main:create_app --factory --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

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
- **Retrieval quality is unmeasured.** There is no evaluation dataset yet, so no accuracy is
  claimed, and the default relevance thresholds (`0.55` for bge-small, `0.10` for hashing) are
  uncalibrated starting points.
- Retrieval is semantic only (no hybrid keyword + vector ranking yet).
- The `hashing` embedder is lexical: synonyms are missed and generic words can match unrelated
  passages. The evidence threshold plus the LLM/citation check are the safeguards.
- Scanned PDFs: pages without a text layer are reported in `empty_pages`; there is no OCR.
- Heading detection is heuristic; `section` is `null` when no heading is found.
- Cost figures are synthetic. There is no trained model and no evaluation metric yet.
- Not implemented: bill auditing, self-healing retrieval loop, authentication, frontend.
