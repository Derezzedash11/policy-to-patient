"""Run the retrieval evaluation.

    python -m evaluation.run --embedder hashing
    python -m evaluation.run --embedder fastembed --json results.json
    python -m evaluation.run --embedder hashing --database-url postgresql://...  # pgvector store

Results describe retrieval on the FICTIONAL evaluation policy only.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from app.config import Settings
from app.embeddings import build_embedder
from app.retrieval.vector import VectorRetriever
from app.vectorstore import build_vector_store
from evaluation.dataset import DEFAULT_DATASET, load_dataset, validate_labels
from evaluation.metrics import K_VALUES, Report, evaluate


def format_report(report: Report) -> str:
    s = report.summary()
    low, high, bal = report.best_thresholds()
    lines = [
        f"## Retrieval evaluation: {report.dataset}",
        "",
        f"- Retriever: `{report.retriever}` ({'semantic' if report.is_semantic else 'LEXICAL, not semantic'})",
        f"- Questions: {s['questions']} ({s['answerable']} answerable, {s['unanswerable']} unanswerable)",
        f"- top_k = {report.top_k}, min_score = {report.min_score}, query rewrites = {report.max_rewrites}",
        "",
        "| Answerable subset | n | " + " | ".join(f"hit@{k}" for k in K_VALUES) + " | keyword hit@5 | MRR@5 |",
        "|---|---|" + "---|" * (len(K_VALUES) + 2),
    ]
    for key in ("all_answerable", "direct", "paraphrase"):
        m = s[key]
        lines.append(
            f"| {key} | {m['n']} | " + " | ".join(f"{m[f'hit@{k}']:.2f}" for k in K_VALUES)
            + f" | {m['keyword_hit@5']:.2f} | {m['mrr@5']:.2f} |"
        )
    g = s["answer_gate"]
    lines += [
        "",
        f"Answer gate at min_score {report.min_score} (pipeline status, no LLM unless configured):",
        f"- answerable questions that passed the gate: {g['answerable_passed']:.2f}",
        f"- answerable questions whose labelled passage was among the evidence sent: "
        f"{g['answerable_with_correct_evidence_sent']:.2f}",
        f"- unanswerable questions rejected as insufficient_evidence: {g['unanswerable_rejected']:.2f}",
        "",
        f"Best top-1 score threshold on this set: {low:.2f}–{high:.2f} "
        f"(balanced accuracy {bal:.2f}; measured on the same questions, so optimistic).",
        "",
        "Misses (answerable questions without a labelled passage in the top k):",
    ]
    misses = [r for r in report.answerable() if r.hit_rank is None]
    lines += [f"- {r.id} ({r.kind}) top score {r.top_score}" for r in misses] or ["- none"]
    passed_unanswerable = [r for r in report.unanswerable if r.status != "insufficient_evidence"]
    lines += ["", "Unanswerable questions that passed the gate:"]
    lines += [f"- {r.id} top score {r.top_score}" for r in passed_unanswerable] or ["- none"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--embedder", default="hashing", choices=["hashing", "fastembed"])
    parser.add_argument("--model", default=None, help="fastembed model name")
    parser.add_argument("--database-url", default=None, help="use pgvector instead of a temp local store")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=None, help="default: the embedder's own")
    parser.add_argument("--json", type=Path, default=None, help="also write full results as JSON")
    args = parser.parse_args(argv)

    dataset = load_dataset(args.dataset)
    problems = validate_labels(dataset, dataset.chunks())
    if problems:
        print("Dataset labels are invalid:\n" + "\n".join(problems), file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            data_dir=Path(tmp), embedding_provider=args.embedder, database_url=args.database_url,
            **({"embedding_model": args.model} if args.model else {}),
        )
        retriever = VectorRetriever(build_embedder(settings), build_vector_store(settings))
        min_score = args.min_score if args.min_score is not None else retriever.default_min_score
        report = evaluate(dataset, retriever, min_score=min_score, top_k=args.top_k)

    print(format_report(report))
    if args.json:
        args.json.write_text(json.dumps(report.to_dict(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
