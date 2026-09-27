# Phase 1 Plan — Policy-to-Patient (HackMatrix 5.0 FIN-01)

Target: a **30–40% working prototype** of

`Policy PDF → text extraction → page-aware chunks → evidence retrieval → policy Q&A with citations → treatment input → cost estimate → deterministic coverage/OOP calculation`

---

## 1. Current architecture

There is none yet. Audit of the repository (branch `main`, 3 commits):

| Item | Status |
|---|---|
| `README.md` | 2-line description |
| `PROJECT_CONTEXT.md` | Full product spec, scope rules, tech direction (Next.js/TS, FastAPI, PostgreSQL + pgvector, scikit-learn) |
| `.gitignore` | Python/Node ignores, `.env` excluded |
| Source code (frontend/backend/ml) | **None** |
| Dependency files (`pyproject.toml`, `requirements.txt`, `package.json`) | **None** |
| Database schema / migrations | **None** |
| Data (sample policy, cost dataset) | **None** |
| Tests / CI | **None** |

Environment observed: Python 3.11, Node 22, `psql` and `docker` binaries available.

## 2. Current implementation status

Nothing is implemented. The only reusable asset is the spec in `PROJECT_CONTEXT.md`, which Phase 1 follows (folder layout, separation of concerns, "never invent policy facts / metrics").

## 3. Gaps / problems

1. **No code, no dependency manifest, no tests.** Everything is greenfield.
2. **No policy PDF.** A real, publicly available policy wording is needed for the demo. It must not be fabricated; the user supplies it (and confirms it may be committed, otherwise it stays local under a git-ignored folder).
3. **No treatment-cost dataset.** The spec requires an ML model trained on a *provided* dataset with real metrics. None exists, so Phase 1 cannot train a model or report metrics. Phase 1 uses a **clearly labelled synthetic demo cost table** behind an interface the ML model will replace.
4. **Policy terms for the rules engine have no source yet.** Automatically extracting deductible/co-pay/limits from free text reliably is out of scope for Phase 1. Terms are entered/confirmed as structured input, each field optionally linked to a cited chunk (page + section). Missing terms are reported as missing, never defaulted.
5. **Infrastructure weight.** PostgreSQL + pgvector plus an embedding model is significant setup for a 30–40% prototype. Phase 1 keeps retrieval behind an interface and uses an in-process lexical index; pgvector/embeddings come in Phase 2 without changing callers.

## 4. Proposed Phase-1 architecture

```
backend/app/
  ingestion/   PDF → pages → chunks            (pypdf)
  retrieval/   Retriever interface + TF-IDF     (scikit-learn)      ← finds evidence only
  qa/          Prompt + LLM client + citation check (anthropic)     ← explains evidence only
  cost/        CostEstimator interface + synthetic demo table        ← estimates cost only
  rules/       Deterministic coverage/OOP engine (pure Python)      ← applies policy terms only
  api/         FastAPI routes
  store.py     Local persistence (JSON files under data/runtime/, git-ignored)
```

Responsibilities stay separate; no module calls the LLM except `qa/`, and the LLM never computes money.

### 4.1 Ingestion
- `pypdf` extracts text **per page** (1-based page numbers kept). No OCR: pages with no text layer are flagged `empty_text=true`.
- Chunker: split on detected headings (numbered clauses / ALL-CAPS lines) then by size (~800 chars, ~150 overlap), never crossing a page boundary. Each chunk:
  `{chunk_id, doc_id, doc_name, page, section (nullable), text, char_start, char_end}`.

### 4.2 Retrieval
- `Retriever` protocol: `index(chunks)`, `search(query, k) -> [ScoredChunk]`.
- Phase-1 implementation: TF-IDF (word + bigram) cosine similarity. Returns scores so a minimum-evidence threshold can trigger "insufficient evidence".
- Phase 2: embeddings + pgvector implementing the same protocol.

### 4.3 Policy Q&A (grounded)
- Retrieve top-k chunks → prompt the LLM with only those chunks, each tagged `[C#]`, instructing it to answer only from them and cite `[C#]`.
- **Post-check (deterministic):** every cited id must be in the retrieved set; answers with no valid citation, or when retrieval score is below threshold, return `status: "insufficient_evidence"` plus the raw evidence.
- Response: `{answer, status, citations: [{doc_name, page, section, quote}], retrieved: [...]}`.
- **No API key → extractive mode:** returns the top evidence passages with citations and no generated text. The app is demoable and testable offline; nothing is faked.
- Single LLM provider (Anthropic SDK). No LangChain/LlamaIndex.

### 4.4 Treatment input + cost estimate
- Input: `{treatment (from a fixed list), city_tier, room_type, length_of_stay_days}`.
- `CostEstimator` interface returns `{estimate, low, high, method, is_synthetic, source}`.
- Phase-1 implementation reads `data/sample/demo_treatment_costs.csv`, whose header and every API response mark it **SYNTHETIC — for demo only, not real prices**. No accuracy metrics are produced.
- Phase 2: scikit-learn model trained on a real provided dataset, metrics computed from held-out data.

### 4.5 Rules engine (deterministic)
- Input `PolicyTerms` (all fields optional, each may carry a `citation`): `sum_insured`, `deductible`, `copay_pct`, `room_rent_limit_per_day`, `sub_limits{category: amount}`, `waiting_period_months` + `policy_age_months`, `excluded_categories`.
- Output: itemised steps (`step, rule, amount_before, amount_after, citation`), `insurer_pays`, `patient_pays`, and `missing_terms[]` / `assumptions[]`.
- Fixed, documented order: exclusion/waiting-period check → sub-limit → room-rent cap → deductible → co-pay → sum-insured cap. Order is a stated assumption, not a claim about any specific insurer.
- All output labelled as an **estimate, not an insurance decision**.

### 4.6 API (FastAPI)
| Method | Path | Purpose |
|---|---|---|
| POST | `/policies` | upload PDF → ingest + index; returns doc_id, page/chunk counts |
| GET | `/policies/{id}/chunks` | inspect chunks (debug / evidence view) |
| POST | `/policies/{id}/ask` | grounded Q&A with citations |
| GET | `/treatments` | list treatments in demo cost table |
| POST | `/estimate` | treatment input → cost estimate (+ synthetic flag) |
| POST | `/coverage` | cost estimate + `PolicyTerms` → deterministic breakdown |
| GET | `/health` | liveness + whether LLM mode is `generative` or `extractive` |

### 4.7 Frontend
Phase 1 demo runs via FastAPI's built-in `/docs` UI. A minimal Next.js page (upload → ask → estimate → coverage) is the **last** Phase-1 step and optional if time is short. No UI component libraries.

## 5. Files to create / modify

Create:
```
backend/pyproject.toml                 # deps + pytest config
backend/.env.example                   # ANTHROPIC_API_KEY, LLM_MODEL, DATA_DIR
backend/app/__init__.py
backend/app/main.py                    # FastAPI app + routers
backend/app/config.py
backend/app/models.py                  # Pydantic schemas (Chunk, Citation, PolicyTerms, ...)
backend/app/store.py
backend/app/ingestion/pdf_extract.py
backend/app/ingestion/chunker.py
backend/app/retrieval/base.py
backend/app/retrieval/tfidf.py
backend/app/qa/prompt.py
backend/app/qa/llm.py
backend/app/qa/answer.py               # retrieve → LLM/extractive → citation check
backend/app/cost/base.py
backend/app/cost/demo_table.py
backend/app/rules/coverage.py
backend/app/api/routes.py
backend/tests/conftest.py              # builds a tiny text PDF fixture in-test (no binary committed)
backend/tests/test_pdf_extract.py
backend/tests/test_chunker.py
backend/tests/test_retrieval.py
backend/tests/test_answer_citations.py # uses a stub LLM; no network
backend/tests/test_cost_demo.py
backend/tests/test_coverage_rules.py
data/sample/demo_treatment_costs.csv   # SYNTHETIC, labelled in header + README
data/sample/README.md                  # provenance of every sample file
```
Modify:
```
README.md      # setup, run, demo flow, "synthetic data" disclaimer
.gitignore     # add data/runtime/, .pytest_cache/
```
Later (end of Phase 1, optional): `frontend/` minimal Next.js app.

## 6. Dependencies

Backend (runtime): `fastapi`, `uvicorn`, `python-multipart` (uploads), `pydantic` (via FastAPI), `pypdf`, `scikit-learn` (TF-IDF now, cost model later), `anthropic` (optional at runtime — extractive mode without key).
Backend (dev): `pytest`, `httpx` (FastAPI TestClient).

Deliberately **not** added in Phase 1: LangChain/LlamaIndex, a second LLM SDK, sentence-transformers/torch, PostgreSQL/pgvector, OCR libraries, auth, Docker/infra.

## 7. Phase-1 acceptance criteria

1. `pip install -e backend[dev]` and `pytest` pass offline with no API key.
2. Uploading a text-based policy PDF returns page count and chunk count; every chunk has `page` (1-based) and `section` when detectable.
3. `/ask` returns citations with `doc_name`, `page`, `section`, `quote`; every citation maps to a retrieved chunk (enforced in code + tested).
4. Low-relevance or uncited answers return `insufficient_evidence` rather than an unsupported answer.
5. Without an API key, `/ask` works in extractive mode and says so.
6. `/estimate` returns a range with `is_synthetic: true` and a source note; no accuracy metrics anywhere.
7. `/coverage` is fully deterministic: identical inputs → identical output; itemised steps; missing terms listed, not guessed; unit tests cover deductible, co-pay, sub-limit, room-rent cap, sum-insured cap, exclusion and waiting period.
8. README documents the demo flow end to end and the synthetic-data disclaimer.

Out of scope for Phase 1: bill upload/auditing, ML training, OCR, self-healing retrieval loop, auth, production infrastructure.

## 8. Implementation order

1. Backend skeleton: `pyproject.toml`, `main.py`, `config.py`, `models.py`, `/health`, pytest wiring.
2. Ingestion: `pdf_extract.py` + `chunker.py` + tests.
3. Retrieval: `base.py` + `tfidf.py` + tests.
4. Rules engine: `rules/coverage.py` + tests (independent of everything else).
5. Cost: synthetic CSV + `cost/demo_table.py` + tests.
6. Q&A: prompt, LLM client, citation checker, extractive fallback + tests with stub LLM.
7. API routes + `store.py`; end-to-end test via TestClient.
8. README update; optional minimal frontend.

## 9. Open decisions for the user

- Which real policy PDF to use for the demo, and may it be committed?
- Is there a treatment-cost dataset for Phase 2 (source, licence, columns)?
- LLM model choice/key availability (plan assumes Anthropic API, optional).
