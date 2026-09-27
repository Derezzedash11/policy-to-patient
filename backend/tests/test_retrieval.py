import pytest

from app.embeddings.hashing import HashingEmbedder
from app.models import Chunk
from app.retrieval.base import NotIndexedError
from app.retrieval.vector import embedding_text
from conftest import ConceptEmbedder, CountingEmbedder

DOC = "testdoc"


def test_index_then_search_returns_page_and_section(indexed_retriever):
    assert indexed_retriever.is_indexed(DOC)
    hits = indexed_retriever.search(DOC, "What is the deductible?", top_k=3)
    top = hits[0].chunk
    assert "deductible of Rs 10,000" in top.text
    assert top.page == 4
    assert top.section == "3. DEDUCTIBLE AND CO-PAYMENT"
    assert top.doc_name == "fictional_policy.pdf"


def test_finds_room_rent_limit(indexed_retriever):
    hits = indexed_retriever.search(DOC, "room rent limit per day", top_k=3)
    assert hits[0].chunk.page == 2
    assert "Rs 5,000 per day" in hits[0].chunk.text


def test_scores_sorted_bounded_and_top_k_respected(indexed_retriever):
    hits = indexed_retriever.search(DOC, "waiting period maternity", top_k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score
    assert all(-1 <= h.score <= 1 for h in hits)


def test_unrelated_query_scores_below_default_threshold(indexed_retriever):
    hits = indexed_retriever.search(DOC, "xylophone quantum volcano", top_k=3)
    assert all(h.score < HashingEmbedder.default_min_score for h in hits)


def test_empty_or_stopword_query_returns_nothing(indexed_retriever):
    assert indexed_retriever.search(DOC, "   ", 3) == []
    assert indexed_retriever.search(DOC, "is it the", 3) == []


def test_unindexed_document_raises(indexed_retriever):
    assert not indexed_retriever.is_indexed("otherdoc")
    with pytest.raises(NotIndexedError):
        indexed_retriever.search("otherdoc", "deductible", 3)


def test_documents_are_isolated(indexed_retriever):
    other = Chunk(
        chunk_id="otherdoc:p1:c0", doc_id="otherdoc", doc_name="other.pdf", page=1,
        section=None, text="The deductible for this other policy is Rs 99,999.", char_start=0, char_end=50,
    )
    indexed_retriever.index("otherdoc", [other])
    mine = indexed_retriever.search(DOC, "deductible", top_k=10)
    theirs = indexed_retriever.search("otherdoc", "deductible", top_k=10)
    assert {h.chunk.doc_id for h in mine} == {DOC}
    assert [h.chunk.chunk_id for h in theirs] == ["otherdoc:p1:c0"]


def test_reindex_replaces_previous_chunks(indexed_retriever, policy_chunks):
    indexed_retriever.index(DOC, policy_chunks[:1])
    hits = indexed_retriever.search(DOC, "deductible co-payment waiting", top_k=10)
    assert [h.chunk.chunk_id for h in hits] == [policy_chunks[0].chunk_id]


def test_semantic_retrieval_matches_meaning_not_words(make_retriever, policy_chunks):
    # The question shares no content words (after stemming) with the maternity passage.
    question = "When is childbirth paid for?"
    semantic = make_retriever(ConceptEmbedder())
    semantic.index(DOC, policy_chunks)
    top = semantic.search(DOC, question, top_k=1)[0]
    assert "waiting period of 24 months" in top.chunk.text
    assert top.score >= ConceptEmbedder.default_min_score

    lexical = make_retriever(HashingEmbedder(), root=None)
    lexical.index("lexdoc", policy_chunks)
    lexical_top = lexical.search("lexdoc", question, top_k=1)
    assert not lexical_top or lexical_top[0].score < HashingEmbedder.default_min_score
    assert "maternity" in top.chunk.text.lower()


def test_section_heading_is_embedded_with_the_chunk():
    chunk = Chunk(chunk_id="d:p1:c0", doc_id="d", doc_name="x", page=1, section="5. EXCLUSIONS",
                  text="Dental treatment.", char_start=0, char_end=17)
    assert embedding_text(chunk) == "5. EXCLUSIONS\nDental treatment."
    headed = chunk.model_copy(update={"text": "5. EXCLUSIONS\nDental treatment."})
    assert embedding_text(headed) == headed.text


def test_index_embeds_each_chunk_once(make_retriever, policy_chunks):
    counting = CountingEmbedder(HashingEmbedder())
    retriever = make_retriever(counting)
    retriever.index(DOC, policy_chunks)
    retriever.search(DOC, "deductible", 3)
    retriever.search(DOC, "room rent", 3)
    assert counting.documents_embedded == len(policy_chunks)
