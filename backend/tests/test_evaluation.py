"""The evaluation framework: dataset integrity, metric arithmetic, and an end-to-end run."""

import json

import pytest

from evaluation.dataset import Dataset, Question, load_dataset, validate_labels
from evaluation.metrics import QuestionResult, Report, evaluate
from evaluation.run import main


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return load_dataset()


@pytest.fixture(scope="module")
def chunks(dataset):
    return dataset.chunks()


def test_dataset_size_and_mix(dataset):
    kinds = [q.kind for q in dataset.questions]
    assert 30 <= len(kinds) <= 50
    assert {"direct", "paraphrase", "unanswerable"} == set(kinds)
    assert "FICTIONAL" in dataset.policy.pages[0][0]
    assert "SYNTHETIC" in dataset.description


def test_every_label_is_consistent_with_the_ingested_policy(dataset, chunks):
    # Answerable: keyword found in the labelled page+section chunk. Unanswerable: absent terms absent.
    assert validate_labels(dataset, chunks) == []


def test_label_validation_catches_mistakes(dataset, chunks):
    bad = dataset.model_copy(deep=True)
    bad.questions[0].keyword = "Rs 9,99,999"
    bad.questions[1].expected[0].section = "99. NOT A SECTION"
    bad.questions.append(Question(id="x1", kind="unanswerable", question="?", answerable=False,
                                  absent_terms=["deductible"]))
    problems = validate_labels(bad, chunks)
    assert any("Rs 9,99,999" in p for p in problems)
    assert any("99. NOT A SECTION" in p for p in problems)
    assert any("'deductible' appears" in p for p in problems)


def _row(id, answerable, top, hit=None, status="evidence_only"):
    return QuestionResult(id=id, kind="direct" if answerable else "unanswerable", answerable=answerable,
                          top_score=top, hit_rank=hit, keyword_rank=hit, status=status,
                          correct_evidence_sent=bool(hit))


def test_metric_arithmetic():
    report = Report("d", "r", False, 0.2, 5, 0, [
        _row("a", True, 0.9, hit=1), _row("b", True, 0.5, hit=3), _row("c", True, 0.1, hit=None,
                                                                          status="insufficient_evidence"),
        _row("u", False, 0.05, status="insufficient_evidence"), _row("v", False, 0.4),
    ])
    s = report.summary()["all_answerable"]
    assert (s["hit@1"], s["hit@3"], s["hit@5"]) == (0.333, 0.667, 0.667)
    assert s["mrr@5"] == round((1 + 1 / 3) / 3, 3)
    gate = report.summary()["answer_gate"]
    assert gate["answerable_passed"] == 0.667 and gate["unanswerable_rejected"] == 0.5
    low, high, bal = report.best_thresholds()
    # Thresholds in (0.4, 0.5] pass a, b and reject u, v → balanced accuracy (2/3 + 1) / 2
    assert (low, high) == (0.41, 0.5) and bal == pytest.approx((2 / 3 + 1) / 2)


def test_end_to_end_evaluation_with_hashing(dataset, make_retriever):
    report = evaluate(dataset, make_retriever(), min_score=0.10)
    summary = report.summary()
    assert summary["questions"] == len(dataset.questions)
    # Regression floor for the lexical baseline on questions that share wording with the policy.
    # Not an accuracy claim: the policy and questions are fictional and hand-written.
    assert summary["direct"]["hit@5"] == 1.0
    assert all(r.queries == [dataset_q.question] for r, dataset_q in zip(report.results, dataset.questions))


def test_cli_prints_report_and_writes_json(tmp_path, capsys):
    out = tmp_path / "eval.json"
    assert main(["--embedder", "hashing", "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "LEXICAL, not semantic" in printed and "hit@1" in printed
    data = json.loads(out.read_text())
    assert data["summary"]["questions"] == 48 and len(data["results"]) == 48


def test_demo_pdf_writes_the_fictional_policy(tmp_path):
    from app.ingestion.pdf_extract import extract_pages
    from evaluation.demo_pdf import main as write_pdf

    out = tmp_path / "demo" / "policy.pdf"
    assert write_pdf([str(out)]) == 0
    pages = extract_pages(out.read_bytes())
    assert len(pages) == 8 and pages[0].text.startswith("FICTIONAL TEST POLICY")
