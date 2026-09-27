"""Retrieval and answer-gating metrics computed over an evaluation dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from app.qa.answer import answer_question
from app.qa.llm import LLMClient
from app.retrieval.base import Retriever
from evaluation.dataset import EVAL_DOC_ID, Dataset, Question, is_hit

K_VALUES = (1, 3, 5)
GATE_PASS = {"evidence_only", "answered", "manual_review"}


@dataclass
class QuestionResult:
    id: str
    kind: str
    answerable: bool
    top_score: float | None
    hit_rank: int | None  # 1-based rank of the first chunk at a labelled page+section
    keyword_rank: int | None  # 1-based rank of the first chunk containing the keyword
    status: str
    correct_evidence_sent: bool  # labelled chunk is among the passages that pass the threshold
    queries: list[str] = field(default_factory=list)


@dataclass
class Report:
    dataset: str
    retriever: str
    is_semantic: bool
    min_score: float
    top_k: int
    max_rewrites: int
    results: list[QuestionResult]

    def answerable(self, kind: str | None = None) -> list[QuestionResult]:
        return [r for r in self.results if r.answerable and (kind is None or r.kind == kind)]

    @property
    def unanswerable(self) -> list[QuestionResult]:
        return [r for r in self.results if not r.answerable]

    @staticmethod
    def hit_rate(rows: list[QuestionResult], k: int, attr: str = "hit_rank") -> float:
        if not rows:
            return 0.0
        return sum(1 for r in rows if (rank := getattr(r, attr)) is not None and rank <= k) / len(rows)

    @staticmethod
    def mrr(rows: list[QuestionResult]) -> float:
        return sum(1 / r.hit_rank for r in rows if r.hit_rank) / len(rows) if rows else 0.0

    def summary(self) -> dict:
        ans, una = self.answerable(), self.unanswerable
        out: dict = {
            "questions": len(self.results),
            "answerable": len(ans),
            "unanswerable": len(una),
        }
        for kind in (None, "direct", "paraphrase"):
            rows = self.answerable(kind)
            key = kind or "all_answerable"
            out[key] = {f"hit@{k}": round(self.hit_rate(rows, k), 3) for k in K_VALUES}
            out[key]["keyword_hit@5"] = round(self.hit_rate(rows, 5, "keyword_rank"), 3)
            out[key]["mrr@5"] = round(self.mrr(rows), 3)
            out[key]["n"] = len(rows)
        out["answer_gate"] = {
            "answerable_passed": round(sum(r.status in GATE_PASS for r in ans) / len(ans), 3) if ans else 0,
            "answerable_with_correct_evidence_sent": round(
                sum(r.correct_evidence_sent for r in ans) / len(ans), 3
            ) if ans else 0,
            "unanswerable_rejected": round(
                sum(r.status == "insufficient_evidence" for r in una) / len(una), 3
            ) if una else 0,
        }
        return out

    def threshold_sweep(self, step: float = 0.01) -> list[dict]:
        """Top-1 score gate at each threshold: share of answerable passing / unanswerable rejected."""
        ans = [r.top_score or 0.0 for r in self.answerable()]
        una = [r.top_score or 0.0 for r in self.unanswerable]
        rows = []
        for i in range(int(round(1 / step)) + 1):
            t = round(i * step, 4)
            tpr = sum(s >= t for s in ans) / len(ans) if ans else 0.0
            tnr = sum(s < t for s in una) / len(una) if una else 0.0
            rows.append({"threshold": t, "answerable_pass": tpr, "unanswerable_reject": tnr,
                         "balanced_accuracy": (tpr + tnr) / 2})
        return rows

    def best_thresholds(self) -> tuple[float, float, float]:
        """(low, high, balanced_accuracy): the range of thresholds with the best balanced accuracy."""
        sweep = self.threshold_sweep()
        best = max(r["balanced_accuracy"] for r in sweep)
        ts = [r["threshold"] for r in sweep if r["balanced_accuracy"] == best]
        return min(ts), max(ts), best

    def to_dict(self) -> dict:
        low, high, bal = self.best_thresholds()
        return {
            "dataset": self.dataset, "retriever": self.retriever, "is_semantic": self.is_semantic,
            "min_score": self.min_score, "top_k": self.top_k, "max_rewrites": self.max_rewrites,
            "summary": self.summary(),
            "best_threshold_range": {"low": low, "high": high, "balanced_accuracy": bal},
            "results": [asdict(r) for r in self.results],
        }


def _rank(items: list, predicate) -> int | None:
    return next((i + 1 for i, item in enumerate(items) if predicate(item)), None)


def evaluate_question(
    q: Question, retriever: Retriever, min_score: float, top_k: int,
    llm: LLMClient | None, max_rewrites: int,
) -> QuestionResult:
    hits = retriever.search(EVAL_DOC_ID, q.question, top_k)
    chunks = [h.chunk for h in hits]
    keyword = (q.keyword or "").lower()
    response = answer_question(q.question, EVAL_DOC_ID, retriever, llm, top_k=top_k,
                               min_score=min_score, max_rewrites=max_rewrites)
    sent_ids = {e.chunk_id for e in response.evidence} if response.status in GATE_PASS else set()
    return QuestionResult(
        id=q.id,
        kind=q.kind,
        answerable=q.answerable,
        top_score=hits[0].score if hits else None,
        hit_rank=_rank(chunks, lambda c: is_hit(c, q)) if q.answerable else None,
        keyword_rank=_rank(chunks, lambda c: keyword in c.text.lower()) if keyword else None,
        status=response.status,
        correct_evidence_sent=q.answerable and any(
            is_hit(c, q) for c in chunks if c.chunk_id in sent_ids
        ),
        queries=[a.query for a in response.retrieval_attempts],
    )


def evaluate(
    dataset: Dataset, retriever: Retriever, min_score: float, top_k: int = 5,
    llm: LLMClient | None = None, max_rewrites: int = 0,
) -> Report:
    retriever.index(EVAL_DOC_ID, dataset.chunks())
    results = [
        evaluate_question(q, retriever, min_score, top_k, llm, max_rewrites) for q in dataset.questions
    ]
    return Report(
        dataset=dataset.name, retriever=retriever.name, is_semantic=retriever.is_semantic,
        min_score=min_score, top_k=top_k, max_rewrites=max_rewrites, results=results,
    )
