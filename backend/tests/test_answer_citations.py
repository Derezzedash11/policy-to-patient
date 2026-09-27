import pytest

from app.qa.answer import answer_question, extract_citation_labels
from app.qa.llm import LLMError
from app.qa.prompt import INSUFFICIENT_MARKER
from app.retrieval.tfidf import TfidfRetriever
from conftest import StubLLM

MIN_SCORE = 0.08


@pytest.fixture
def retriever(policy_chunks):
    r = TfidfRetriever()
    r.index(policy_chunks)
    return r


def ask(retriever, llm, question="What is the deductible each policy year?"):
    return answer_question(question, retriever, llm, top_k=3, min_score=MIN_SCORE)


def test_extract_citation_labels():
    assert extract_citation_labels("A [C2]. B [C1, C3] and [C2]") == ["C2", "C1", "C3"]
    assert extract_citation_labels("no citations") == []


def test_extractive_mode_returns_evidence_not_an_answer(retriever):
    res = ask(retriever, None)
    assert res.status == "evidence_only"
    assert res.mode == "extractive"
    assert res.answer is None
    assert res.citations and res.citations[0].page == 4
    assert "deductible" in res.citations[0].quote.lower()


def test_valid_citations_are_accepted_and_mapped(retriever):
    llm = StubLLM("A deductible of Rs 10,000 applies each policy year [C1].")
    res = ask(retriever, llm)
    assert res.status == "answered"
    assert res.mode == "generative"
    assert [c.ref for c in res.citations] == ["C1"]
    assert res.citations[0].page == 4
    assert res.citations[0].section == "3. DEDUCTIBLE AND CO-PAYMENT"


def test_llm_only_sees_retrieved_passages(retriever):
    llm = StubLLM("x [C1]")
    res = ask(retriever, llm)
    _, user_prompt = llm.calls[0]
    assert "deductible of Rs 10,000" in user_prompt
    # One labelled block per evidence passage, and nothing else from the document.
    assert user_prompt.count("] page ") == len(res.evidence)
    assert "Hospitalisation means admission" not in user_prompt


def test_citation_to_unretrieved_passage_is_rejected(retriever):
    res = ask(retriever, StubLLM("The deductible is Rs 10,000 [C1] [C9]."))
    assert res.status == "insufficient_evidence"
    assert res.answer is None
    assert "C9" in res.message


def test_uncited_answer_is_rejected(retriever):
    res = ask(retriever, StubLLM("The deductible is Rs 10,000."))
    assert res.status == "insufficient_evidence"
    assert res.answer is None


def test_llm_declares_insufficient(retriever):
    res = ask(retriever, StubLLM(INSUFFICIENT_MARKER))
    assert res.status == "insufficient_evidence"
    assert res.answer is None
    assert res.evidence  # evidence still shown for manual review


def test_low_relevance_question_is_insufficient_without_calling_llm(retriever):
    llm = StubLLM("should not be used [C1]")
    res = ask(retriever, llm, "Does the policy cover space tourism?")
    assert res.status == "insufficient_evidence"
    assert llm.calls == []


def test_threshold_is_configurable(retriever):
    res = answer_question("deductible", retriever, None, top_k=3, min_score=0.99)
    assert res.status == "insufficient_evidence"
    assert res.evidence  # below-threshold hits returned for transparency


def test_llm_error_falls_back_to_evidence(retriever):
    class Failing(StubLLM):
        def complete(self, system, user):
            raise LLMError("LLM API error 500: boom")

    res = ask(retriever, Failing(""))
    assert res.status == "llm_error"
    assert res.answer is None
    assert res.citations
