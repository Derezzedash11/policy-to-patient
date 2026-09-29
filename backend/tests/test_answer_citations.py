import pytest

from app.qa.answer import answer_question, extract_citation_labels
from app.qa.llm import LLMError
from app.qa.prompt import INSUFFICIENT_MARKER
from conftest import ConceptEmbedder, ScriptedLLM, StubLLM

MIN_SCORE = 0.10
DOC = "testdoc"


@pytest.fixture
def retriever(indexed_retriever):
    return indexed_retriever


def ask(retriever, llm, question="What is the deductible each policy year?"):
    return answer_question(question, DOC, retriever, llm, top_k=3, min_score=MIN_SCORE)


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
    assert res.status == "manual_review"
    assert res.answer is None
    assert any("C9" in issue for issue in res.verification_issues)
    assert res.citations  # evidence is still shown for manual review


def test_uncited_answer_is_rejected(retriever):
    res = ask(retriever, StubLLM("The deductible is Rs 10,000."))
    assert res.status == "manual_review"
    assert res.answer is None
    assert "Does not cite any provided passage" in res.verification_issues


def test_llm_declares_insufficient(retriever):
    res = ask(retriever, StubLLM(INSUFFICIENT_MARKER))
    assert res.status == "insufficient_evidence"
    assert res.answer is None
    assert res.evidence  # evidence still shown for manual review


def test_low_relevance_question_is_insufficient_without_calling_llm(retriever):
    llm = StubLLM("should not be used [C1]")
    res = answer_question("Does the policy cover space tourism?", DOC, retriever, llm,
                          top_k=3, min_score=MIN_SCORE, max_rewrites=0)
    assert res.status == "insufficient_evidence"
    assert llm.calls == []
    assert [a.strategy for a in res.retrieval_attempts] == ["original"]


def test_threshold_is_configurable(retriever):
    res = answer_question("deductible", DOC, retriever, None, top_k=3, min_score=0.99)
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


def test_grounded_answer_from_semantic_retrieval(make_retriever, policy_chunks):
    # Paraphrased question → semantic stand-in retrieves the waiting-period passage → the LLM
    # sees only retrieved passages and its citation maps back to page 4.
    retriever = make_retriever(ConceptEmbedder())
    retriever.index(DOC, policy_chunks)
    llm = StubLLM("Pregnancy costs are covered after a 24-month waiting period [C1].")
    res = answer_question(
        "How long must I wait before pregnancy costs are paid?", DOC, retriever, llm,
        top_k=3, min_score=ConceptEmbedder.default_min_score,
    )
    assert res.status == "answered"
    assert res.citations[0].page == 4
    assert res.citations[0].section == "4. WAITING PERIODS"
    _, user_prompt = llm.calls[0]
    assert "waiting period of 24 months" in user_prompt
    assert "Room rent" not in user_prompt  # below-threshold passages are never sent


def test_only_evidence_above_threshold_is_sent_to_llm(retriever):
    llm = StubLLM("x [C1]")
    res = answer_question("room rent limit per day", DOC, retriever, llm, top_k=5, min_score=MIN_SCORE)
    _, user_prompt = llm.calls[0]
    assert all(e.score >= MIN_SCORE for e in res.evidence)
    assert user_prompt.count("] page ") == len(res.evidence)


# --------------------------------------------------------------------------- figure verification


def test_answer_with_figure_not_in_cited_passage_needs_manual_review(retriever):
    res = ask(retriever, StubLLM("The deductible is Rs 50,000 per policy year [C1]."))
    assert res.status == "manual_review"
    assert res.answer is None
    assert any("50000 not found in cited passage(s) C1" in i for i in res.verification_issues)


def test_figure_formats_are_normalised(retriever):
    # "10,000" and "10000" are the same figure; "ten" is read as 10.
    res = ask(retriever, StubLLM("You pay the first Rs 10000 each policy year [C1]. A co-payment of ten percent applies [C1]."))
    assert res.status == "answered", res.verification_issues


def test_sentence_with_figure_but_no_citation_needs_manual_review(retriever):
    res = ask(retriever, StubLLM("A deductible applies [C1]. It is Rs 10,000."))
    assert res.status == "manual_review"
    assert any("without citing" in i for i in res.verification_issues)


# --------------------------------------------------------------------------- self-correcting retrieval

PARAPHRASE = "When is childbirth paid for?"  # no content words shared with the maternity passage


def test_rewrite_recovers_evidence_and_answer_uses_original_question(retriever):
    llm = ScriptedLLM("maternity waiting period", "Maternity is covered after 24 months [C1].")
    res = answer_question(PARAPHRASE, DOC, retriever, llm, top_k=3, min_score=MIN_SCORE)
    assert [a.strategy for a in res.retrieval_attempts] == ["original", "llm_rewrite"]
    assert res.retrieval_attempts[0].evidence_found is False
    assert res.retrieval_attempts[1].query == "maternity waiting period"
    assert res.retrieval_attempts[1].evidence_found is True
    assert res.status == "answered"
    assert res.citations[0].section == "4. WAITING PERIODS"
    rewrite_call, answer_call = llm.calls
    assert "Do not answer the question" in rewrite_call[0]
    assert f"Question: {PARAPHRASE}" in answer_call[1]  # the answer is for the user's question
    assert "maternity waiting period" not in answer_call[1]


def test_rewrite_that_still_finds_nothing_returns_insufficient_evidence(retriever):
    llm = ScriptedLLM("astronaut training reimbursement")
    res = answer_question("Is space tourism paid?", DOC, retriever, llm, top_k=3, min_score=MIN_SCORE)
    assert res.status == "insufficient_evidence"
    assert len(llm.calls) == 1  # rewrite only; no answer generated without evidence
    assert len(res.retrieval_attempts) == 2
    assert "after 1 rewritten query" in res.message


def test_rewrites_are_bounded(retriever):
    llm = ScriptedLLM("space one", "space two", "space three", "space four")
    res = answer_question("Is space tourism paid?", DOC, retriever, llm, top_k=3,
                          min_score=MIN_SCORE, max_rewrites=10)
    assert len(res.retrieval_attempts) == 3  # original + hard cap of 2 rewrites
    assert res.status == "insufficient_evidence"


def test_repeated_or_empty_rewrite_stops_the_loop(retriever):
    res = answer_question("Is space tourism paid?", DOC, retriever, ScriptedLLM('"Is space tourism paid?"'),
                          top_k=3, min_score=MIN_SCORE, max_rewrites=2)
    assert res.retrieval_attempts[-1].note == "no new query produced"
    assert res.status == "insufficient_evidence"


def test_rewrite_failure_is_recorded(retriever):
    class Failing(StubLLM):
        def complete(self, system, user):
            raise LLMError("LLM API error 529: overloaded")

    res = answer_question(PARAPHRASE, DOC, retriever, Failing(""), top_k=3, min_score=MIN_SCORE)
    assert res.status == "insufficient_evidence"
    assert "rewrite failed" in res.retrieval_attempts[-1].note


def test_no_rewrite_when_first_retrieval_succeeds(retriever):
    llm = ScriptedLLM("The deductible is Rs 10,000 [C1].")
    res = ask(retriever, llm)
    assert [a.strategy for a in res.retrieval_attempts] == ["original"]
    assert len(llm.calls) == 1


def test_no_rewrite_without_llm(retriever):
    res = answer_question(PARAPHRASE, DOC, retriever, None, top_k=3, min_score=MIN_SCORE, max_rewrites=2)
    assert res.status == "insufficient_evidence"
    assert len(res.retrieval_attempts) == 1
