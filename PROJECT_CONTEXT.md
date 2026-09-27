# Policy-to-Patient — HackMatrix 5.0 FIN-01

## Problem Statement
HackMatrix 5.0 — Finance PS FIN-01:
Policy-to-Patient: Insurance Coverage & Treatment Cost Intelligence.

## Product
A web application that helps users understand an insurance policy in relation to a treatment scenario.

Core flow:

Upload Policy
→ Extract & understand policy
→ Ask grounded questions
→ Enter treatment scenario
→ Estimate treatment cost
→ Calculate potential coverage/OOP
→ Verify evidence
→ Generate report

## Primary MVP

1. Insurance policy PDF upload.
2. PDF text extraction with page numbers.
3. Policy chunking with section/page metadata.
4. Semantic retrieval/RAG.
5. Grounded policy Q&A.
6. Exact page/section evidence for important answers.
7. Treatment scenario input.
8. Treatment cost prediction using an actual ML model.
9. Deterministic coverage/OOP calculation.
10. Confidence/uncertainty handling.
11. Evidence verification/self-healing retrieval loop.

## Advanced MVP

12. Hospital bill upload.
13. Bill line-item extraction.
14. Bill-policy comparison.
15. Potential mismatch/anomaly detection.
16. Evidence-backed review flags.
17. Case summary/report generation.

## AI/ML

The system must contain meaningful AI/ML beyond a basic chatbot.

AI:
- LLM-based policy reasoning.
- RAG.
- Query rewriting.
- Evidence-grounded responses.
- Treatment understanding.

ML:
- Treatment-cost prediction using a provided dataset.
- Evaluation metrics must be calculated from actual data.
- Do not fabricate datasets, metrics, accuracy, or confidence values.
- Optional bill anomaly detection.

Deterministic logic:
- Deductibles.
- Co-pay.
- Sub-limits.
- Coverage limits.
- Waiting periods.
- Other extracted policy constraints.

## Self-Healing Verification

For important answers:

Draft reasoning
→ retrieve evidence
→ verify evidence
→ if insufficient/contradictory:
   reformulate query
   → retrieve again
   → verify again
→ if still insufficient:
   return uncertainty/manual-review state.

The system must never invent policy facts.

## Evidence

Policy evidence must preserve:

- document name
- page number
- section/heading when available
- relevant text/chunk

Important claims should be traceable to the source policy.

## Uncertainty

Clearly distinguish:

- Policy facts
- Model predictions
- Deterministic calculations
- Recommendations/review flags

If required information is missing, state that it is missing rather than guessing.

## Technology Direction

Frontend:
- React / Next.js
- TypeScript

Backend:
- Python
- FastAPI

Database:
- PostgreSQL
- pgvector where appropriate

AI:
- LLM
- Embeddings
- RAG

ML:
- Python
- scikit-learn or suitable ML library

## Suggested Structure

policy-to-patient/
├── frontend/
├── backend/
├── ml/
├── data/
│   └── sample/
├── docs/
├── tests/
├── PROJECT_CONTEXT.md
├── README.md
└── .gitignore

## Development Rules

- Work incrementally.
- Implement one module at a time.
- Do not modify unrelated files.
- Do not add unnecessary dependencies.
- Do not fabricate data or evaluation results.
- Do not hard-code fake AI/ML outputs as final functionality.
- Preserve page/section evidence throughout the pipeline.
- Keep frontend, backend, ML and data layers separated.
- Write tests for important logic.
- Run tests after meaningful changes.
- Do not automatically commit changes.
- Do not create unnecessary commits.
- Keep commits focused on meaningful completed work.

## Implementation Order

1. Project foundation
2. Policy PDF ingestion
3. Policy chunking and metadata
4. Retrieval/vector search
5. Grounded policy Q&A
6. Treatment scenario engine
7. Treatment-cost ML model
8. Coverage/OOP calculation
9. Evidence verification/self-healing
10. Bill extraction
11. Bill-policy comparison
12. Frontend integration
13. Testing
14. Documentation
15. Demo preparation

## Important

This is a hackathon prototype for FIN-01.

Do not represent predictions as guaranteed insurance decisions.
Do not claim fraud from anomaly detection.
Do not invent policy clauses.
Do not invent ML metrics.
Do not implement features outside the agreed scope unless explicitly requested.
