# Phase 2 — Semantic Retrieval + RAG

## Architecture change

```
Phase 1:  chunks ──(rebuilt in memory on every restart)──▶ TF-IDF ──▶ search(query)
Phase 2:  chunks ──embed once at upload──▶ VectorStore (persistent) ──▶ search(doc_id, query)
```

| Interface | Implementations | Selected by |
|---|---|---|
| `Embedder` (`app/embeddings/base.py`) | `FastEmbedEmbedder` (local ONNX `BAAI/bge-small-en-v1.5`, 384-d), `HashingEmbedder` (lexical, 1024-d) | `EMBEDDING_PROVIDER` |
| `VectorStore` (`app/vectorstore/base.py`) | `PgVectorStore` (PostgreSQL + pgvector), `LocalVectorStore` (`.npz` per document) | `DATABASE_URL` set → pgvector |
| `Retriever` (`app/retrieval/base.py`) | `VectorRetriever` = Embedder + VectorStore | — |

- **Index once.** `POST /policies` embeds the chunks and writes the vectors before the policy
  summary is saved. On restart nothing is re-embedded; searches read stored vectors.
- **Keyed by embedding model.** Every vector row carries the embedder name. If the provider or
  model changes, a policy is re-indexed once from its stored chunks on first use, so vectors from
  different models are never compared.
- **Per-policy search.** Every query is filtered by `doc_id` (and model). pgvector uses a
  btree index on `(embedding_model, doc_id)` and an exact cosine scan of that one policy's chunks
  rather than an approximate HNSW index. That keeps results exact and lets models of different
  dimensions share one table.
- **Stored per chunk:** model, doc_id, chunk_id, doc_name, page, section, text, char offsets, vector.
  `policy_index` records which (model, doc_id) pairs are completely indexed (including 0-chunk
  documents); chunks and index row are written in one transaction.
- **Q&A unchanged in contract:** only passages at or above the relevance threshold are sent to
  Claude as `[C#]`; uncited answers or citations to passages that weren't sent are discarded; weak
  retrieval returns `insufficient_evidence`. The threshold defaults to the embedder's own value
  (cosine scales differ by model) and can be overridden with `MIN_EVIDENCE_SCORE`.
- **Removed:** TF-IDF retriever and the scikit-learn dependency (it returns with the cost model).
- **Hybrid retrieval: not added.** It needs a keyword index for each store (Postgres full-text
  search and a separate local implementation) plus score fusion. Deferred until an evaluation
  shows semantic-only retrieval missing exact terms such as clause numbers or amounts.

## Verification performed

- Test suite: vector store contract tests run against both the local store and a real PostgreSQL 16
  + pgvector 0.6 database.
- Live server on PostgreSQL + pgvector: upload → search → ask (answerable and unanswerable) →
  restart → identical search scores, `policy_index.indexed_at` unchanged, no re-index events.

## Not verified here

- **The fastembed model was not run.** The development container's network policy blocks
  Hugging Face, so the model could not be downloaded. The adapter is unit-tested against a fake
  `TextEmbedding`; the live run used the `hashing` embedder. First real run should be on a machine
  that can reach huggingface.co.
- **Generative answers against the live Claude API** (no API key in the container). Covered by
  tests with a stub LLM and by the Phase-1 live check of the error fallback.
- **Retrieval quality.** No evaluation dataset exists, so no accuracy is claimed and the default
  thresholds (0.55 bge-small, 0.10 hashing) are uncalibrated.
