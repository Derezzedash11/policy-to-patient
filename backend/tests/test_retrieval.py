from app.retrieval.tfidf import TfidfRetriever


def _retriever(chunks):
    r = TfidfRetriever()
    r.index(chunks)
    return r


def test_finds_deductible_passage_with_page(policy_chunks):
    hits = _retriever(policy_chunks).search("What is the deductible?", top_k=3)
    assert hits
    top = hits[0].chunk
    assert "deductible" in top.text.lower()
    assert top.page == 4


def test_finds_room_rent_limit(policy_chunks):
    hits = _retriever(policy_chunks).search("room rent limit per day", top_k=3)
    assert hits[0].chunk.page == 2
    assert "Rs 5,000 per day" in hits[0].chunk.text


def test_scores_sorted_and_bounded(policy_chunks):
    hits = _retriever(policy_chunks).search("waiting period maternity", top_k=5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0 < s <= 1 for s in scores)
    assert len(hits) <= 5


def test_unrelated_query_returns_nothing(policy_chunks):
    assert _retriever(policy_chunks).search("xylophone quantum volcano", top_k=5) == []


def test_empty_index_and_empty_query(policy_chunks):
    assert TfidfRetriever().search("deductible", 3) == []
    assert _retriever(policy_chunks).search("   ", 3) == []
