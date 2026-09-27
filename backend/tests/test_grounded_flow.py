"""Full HTTP flow: PDF → chunks → embeddings → vector retrieval → evidence → (mock) Claude → verified answer.

No API key is available in CI, so Claude is replaced by scripted mocks that record exactly what
they were sent. The real AnthropicLLM wrapper is covered separately in test_llm.py.
"""

import re

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings.hashing import HashingEmbedder
from app.main import create_app
from conftest import ScriptedLLM
from evaluation.dataset import load_dataset
from pdf_factory import make_text_pdf

PASSAGE_HEADER = re.compile(r"^\[C(\d+)\] page (\d+)(?:, section: (.*))?$", re.MULTILINE)


class RecordingLLM(ScriptedLLM):
    """Scripted replies; additionally parses the passages it was shown."""

    def passages_shown(self, call: int = -1) -> list[tuple[int, str | None]]:
        return [(int(p), s) for _, p, s in PASSAGE_HEADER.findall(self.calls[call][1])]


@pytest.fixture
def eval_pdf():
    return load_dataset().policy_pdf()


def make(tmp_path, llm, **settings):
    s = Settings(data_dir=tmp_path, anthropic_api_key=None, **settings)
    return TestClient(create_app(s, llm=llm, embedder=HashingEmbedder()))


def upload(client, pdf, name="fictional_health_shield_policy.pdf"):
    r = client.post("/policies", files={"file": (name, pdf, "application/pdf")})
    assert r.status_code == 201, r.text
    return r.json()["doc_id"]


def ask(client, doc_id, question):
    r = client.post(f"/policies/{doc_id}/ask", json={"question": question})
    assert r.status_code == 200, r.text
    return r.json()


def test_claude_receives_only_retrieved_evidence_and_citations_map_back(tmp_path, eval_pdf):
    llm = RecordingLLM("A deductible of Rs 10,000 applies to each policy year [C1].")
    client = make(tmp_path, llm)
    doc_id = upload(client, eval_pdf)
    res = ask(client, doc_id, "What is the deductible for each policy year?")

    assert res["status"] == "answered" and res["mode"] == "generative"
    shown = llm.passages_shown()
    assert shown == [(e["page"], e["section"]) for e in res["evidence"]]  # exactly the evidence list
    assert all(e["score"] >= client.get("/health").json()["min_evidence_score"] for e in res["evidence"])
    prompt = llm.calls[-1][1]
    assert "Cataract surgery is covered" not in prompt  # an unrelated section never reaches Claude
    cite = res["citations"][0]
    assert (cite["ref"], cite["page"], cite["section"]) == ("C1", 4, "8. DEDUCTIBLE")
    assert "Rs 10,000" in cite["quote"]


def test_fabricated_figure_is_withheld_for_manual_review(tmp_path, eval_pdf):
    llm = RecordingLLM("The deductible is Rs 25,000 per year [C1].")
    client = make(tmp_path, llm)
    res = ask(client, upload(client, eval_pdf), "What is the deductible for each policy year?")
    assert res["status"] == "manual_review"
    assert res["answer"] is None
    assert any("25000" in issue for issue in res["verification_issues"])
    assert res["citations"]  # the real passages are shown instead


def test_citation_to_passage_not_supplied_is_withheld(tmp_path, eval_pdf):
    client = make(tmp_path, RecordingLLM("The deductible is Rs 10,000 [C7]."))
    res = ask(client, upload(client, eval_pdf), "What is the deductible for each policy year?")
    assert res["status"] == "manual_review" and res["answer"] is None


def test_unanswerable_question_never_reaches_answer_generation(tmp_path, eval_pdf):
    # This mock would fabricate an answer if it were ever asked to answer.
    llm = RecordingLLM("bariatric obesity procedure", "Bariatric surgery is covered up to Rs 1,00,000 [C1].")
    client = make(tmp_path, llm)
    res = ask(client, upload(client, eval_pdf), "Is bariatric surgery covered?")
    assert res["status"] == "insufficient_evidence"
    assert res["answer"] is None
    assert len(llm.calls) == 1 and "Do not answer the question" in llm.calls[0][0]  # rewrite only
    assert [a["strategy"] for a in res["retrieval_attempts"]] == ["original", "llm_rewrite"]


def test_rewrite_that_reaches_an_unrelated_passage_cannot_smuggle_in_a_fabricated_figure(tmp_path, eval_pdf):
    # Found while building Phase 3: with the lexical embedder, the rewrite "vaccination cover
    # limit" weakly matches the maternity section. The fabricated amount is then caught by the
    # figure check. (An invented claim with no figures would NOT be caught deterministically;
    # that relies on Claude answering INSUFFICIENT_EVIDENCE - see README limitations.)
    llm = RecordingLLM("vaccination cover limit", "Vaccinations are covered up to Rs 5,000 [C1].")
    client = make(tmp_path, llm)
    res = ask(client, upload(client, eval_pdf), "Is there cover for vaccinations?")
    assert res["retrieval_attempts"][0]["evidence_found"] is False
    assert res["status"] == "manual_review" and res["answer"] is None
    assert any("5000 not found" in issue for issue in res["verification_issues"])


def test_llm_saying_insufficient_is_respected(tmp_path, eval_pdf):
    client = make(tmp_path, RecordingLLM("INSUFFICIENT_EVIDENCE"))
    res = ask(client, upload(client, eval_pdf), "What is the deductible for each policy year?")
    assert res["status"] == "insufficient_evidence" and res["answer"] is None


def test_self_correction_through_the_api(tmp_path, eval_pdf):
    llm = RecordingLLM("co-payment percentage", "The insured person bears a co-payment of 10 percent [C1].")
    client = make(tmp_path, llm)
    res = ask(client, upload(client, eval_pdf), "What share of each bill do I have to pay myself?")
    attempts = res["retrieval_attempts"]
    assert [a["strategy"] for a in attempts] == ["original", "llm_rewrite"]
    assert attempts[0]["evidence_found"] is False and attempts[1]["evidence_found"] is True
    assert res["status"] == "answered"
    assert res["citations"][0]["section"] == "9. CO-PAYMENT"


def test_self_correction_can_be_disabled(tmp_path, eval_pdf):
    llm = RecordingLLM()
    client = make(tmp_path, llm, max_query_rewrites=0)
    res = ask(client, upload(client, eval_pdf), "What share of each bill do I have to pay myself?")
    assert res["status"] == "insufficient_evidence" and llm.calls == []
    assert client.get("/health").json()["max_query_rewrites"] == 0


def test_other_policies_never_leak_into_the_prompt(tmp_path, eval_pdf):
    other = make_text_pdf([["OTHER FICTIONAL POLICY", "8. DEDUCTIBLE", "A deductible of Rs 77,777 applies."]])
    llm = RecordingLLM("A deductible of Rs 10,000 applies [C1].")
    client = make(tmp_path, llm)
    doc_id = upload(client, eval_pdf)
    upload(client, other, "other.pdf")
    res = ask(client, doc_id, "What is the deductible for each policy year?")
    assert res["status"] == "answered"
    assert "77,777" not in llm.calls[-1][1]
    assert all(c["chunk_id"].startswith(doc_id) for c in res["citations"])
