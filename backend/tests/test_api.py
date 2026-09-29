import os

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings.hashing import HashingEmbedder
from app.main import create_app
from conftest import ConceptEmbedder, CountingEmbedder, StubLLM
from pdf_factory import make_text_pdf


def make_client(tmp_path, llm=None, embedder=None, **settings):
    settings = Settings(data_dir=tmp_path, anthropic_api_key=None, **settings)
    return TestClient(create_app(settings, llm=llm, embedder=embedder or HashingEmbedder()))


@pytest.fixture
def client(tmp_path):
    return make_client(tmp_path)


def upload(client, pdf):
    return client.post("/policies", files={"file": ("fictional_policy.pdf", pdf, "application/pdf")})


def test_health_reports_extractive_mode(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_mode"] == "extractive"
    assert body["retrieval"] == "hashing-v1:1024 @ local-files"
    assert body["retrieval_is_semantic"] is False
    assert body["min_evidence_score"] == HashingEmbedder.default_min_score


def test_end_to_end_flow(client, policy_pdf):
    # 1. Upload + index
    r = upload(client, policy_pdf)
    assert r.status_code == 201, r.text
    summary = r.json()
    doc_id = summary["doc_id"]
    assert summary["page_count"] == 4
    assert summary["empty_pages"] == [3]
    assert "3. DEDUCTIBLE AND CO-PAYMENT" in summary["sections"]

    # 2. Retrieve policy + chunks
    assert client.get(f"/policies/{doc_id}").json()["doc_id"] == doc_id
    assert [p["doc_id"] for p in client.get("/policies").json()] == [doc_id]
    chunks = client.get(f"/policies/{doc_id}/chunks", params={"page": 2}).json()
    assert chunks and all(c["page"] == 2 for c in chunks)

    # 3. Search + ask (extractive, no key)
    hits = client.post(f"/policies/{doc_id}/search", json={"query": "co-payment"}).json()
    assert hits[0]["page"] == 4
    ask = client.post(f"/policies/{doc_id}/ask", json={"question": "What is the co-payment?"}).json()
    assert ask["status"] == "evidence_only"
    assert ask["answer"] is None
    assert ask["citations"][0]["page"] == 4

    # 4. Treatments + estimate
    treatments = client.get("/treatments").json()
    assert any(t["treatment_code"] == "CATARACT" for t in treatments)
    est = client.post("/estimate", json={"treatment_code": "CATARACT", "city_tier": "tier2"}).json()
    assert est["is_synthetic"] is True

    # 5. Coverage from a treatment, with a term cited back to the retrieved chunk
    deductible_hit = client.post(f"/policies/{doc_id}/search", json={"query": "deductible"}).json()[0]
    cov = client.post(
        "/coverage",
        json={
            "treatment": {"treatment_code": "CATARACT", "city_tier": "tier2"},
            "terms": {
                "sum_insured": 500000, "deductible": 10000, "copay_pct": 10,
                "sub_limits": {"cataract": 40000},
                "citations": {"deductible": {"chunk_id": deductible_hit["chunk_id"],
                                             "page": deductible_hit["page"]}},
            },
        },
    ).json()
    assert cov["status"] == "complete"
    assert cov["cost_estimate"]["is_synthetic"] is True
    assert cov["covered_amount"] + cov["out_of_pocket"] == pytest.approx(cov["total_cost"])
    step = next(s for s in cov["steps"] if s["rule"] == "deductible")
    assert step["citation"]["page"] == 4
    assert any("SYNTHETIC" in a for a in cov["assumptions"])


def test_generative_mode_with_stub_llm(tmp_path, policy_pdf):
    llm = StubLLM("Each policy year has a deductible of Rs 10,000 [C1].")
    client = make_client(tmp_path, llm=llm)
    doc_id = upload(client, policy_pdf).json()["doc_id"]
    assert client.get("/health").json()["llm_mode"] == "generative"
    ask = client.post(f"/policies/{doc_id}/ask", json={"question": "What is the deductible?"}).json()
    assert ask["status"] == "answered"
    assert ask["citations"][0]["page"] == 4


def test_coverage_with_explicit_total(client):
    r = client.post("/coverage", json={"total_cost": 100000, "terms": {"deductible": 10000}})
    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "incomplete_terms"
    assert set(body["missing_terms"]) == {"sum_insured", "copay_pct"}
    assert body["covered_amount"] == 90000


def test_restart_reuses_stored_vectors_without_re_embedding(tmp_path, policy_pdf):
    first_embedder = CountingEmbedder(HashingEmbedder())
    first = make_client(tmp_path, embedder=first_embedder)
    summary = upload(first, policy_pdf).json()
    assert first_embedder.documents_embedded == summary["chunk_count"]  # embedded once at ingestion
    before = first.post(f"/policies/{summary['doc_id']}/search", json={"query": "room rent"}).json()

    restarted_embedder = CountingEmbedder(HashingEmbedder())
    restarted = make_client(tmp_path, embedder=restarted_embedder)
    after = restarted.post(f"/policies/{summary['doc_id']}/search", json={"query": "room rent"}).json()
    ask = restarted.post(f"/policies/{summary['doc_id']}/ask", json={"question": "room rent limit?"}).json()
    assert restarted_embedder.documents_embedded == 0
    assert after == before
    assert ask["status"] == "evidence_only" and ask["citations"][0]["page"] == 2


def test_changing_embedding_model_reindexes_once_from_stored_chunks(tmp_path, policy_pdf):
    doc_id = upload(make_client(tmp_path), policy_pdf).json()["doc_id"]
    concept = CountingEmbedder(ConceptEmbedder())
    client = make_client(tmp_path, embedder=concept)
    question = {"question": "How long must I wait before pregnancy costs are paid?"}
    first = client.post(f"/policies/{doc_id}/ask", json=question).json()
    client.post(f"/policies/{doc_id}/ask", json=question)
    assert concept.documents_embedded == client.get(f"/policies/{doc_id}").json()["chunk_count"]
    assert first["status"] == "evidence_only"
    assert first["citations"][0]["section"] == "4. WAITING PERIODS"


def test_policies_are_isolated(client, policy_pdf):
    doc_a = upload(client, policy_pdf).json()["doc_id"]
    other_pdf = make_text_pdf([["OTHER FICTIONAL POLICY", "A deductible of Rs 25,000 applies."]])
    doc_b = client.post("/policies", files={"file": ("other.pdf", other_pdf, "application/pdf")}).json()["doc_id"]
    assert doc_a != doc_b
    hits_a = client.post(f"/policies/{doc_a}/search", json={"query": "deductible", "top_k": 20}).json()
    hits_b = client.post(f"/policies/{doc_b}/search", json={"query": "deductible", "top_k": 20}).json()
    assert all(h["chunk_id"].startswith(doc_a) for h in hits_a)
    assert all(h["chunk_id"].startswith(doc_b) for h in hits_b)
    assert "Rs 25,000" in hits_b[0]["text"] and all("25,000" not in h["text"] for h in hits_a)


def test_min_evidence_score_override(tmp_path, policy_pdf):
    client = make_client(tmp_path, min_evidence_score=0.99)
    assert client.get("/health").json()["min_evidence_score"] == 0.99
    doc_id = upload(client, policy_pdf).json()["doc_id"]
    ask = client.post(f"/policies/{doc_id}/ask", json={"question": "What is the deductible?"}).json()
    assert ask["status"] == "insufficient_evidence"


def test_error_cases(client):
    assert upload(client, b"not a pdf").status_code == 422
    assert upload(client, b"%PDF-1.7 truncated").status_code == 422
    assert client.get("/policies/unknown123").status_code == 404
    assert client.get("/policies/../etc").status_code == 404
    assert client.post("/policies/unknown123/ask", json={"question": "deductible?"}).status_code == 404
    assert client.post("/estimate", json={"treatment_code": "NOPE"}).status_code == 404
    assert client.post("/coverage", json={"terms": {}}).status_code == 422
    both = {"terms": {}, "total_cost": 50000, "treatment": {"treatment_code": "CATARACT"}}
    r = client.post("/coverage", json=both)
    assert r.status_code == 422 and "not both" in r.json()["detail"]
    assert client.post("/estimate", json={"treatment_code": "CATARACT", "city_tier": "moon"}).status_code == 422


def test_policy_without_text_is_indexed_and_reports_insufficient_evidence(client):
    scanned = make_text_pdf([[], []])  # pages with no text layer, like a scan (no OCR)
    summary = client.post("/policies", files={"file": ("scan.pdf", scanned, "application/pdf")}).json()
    assert summary["chunk_count"] == 0 and summary["empty_pages"] == [1, 2]
    ask = client.post(f"/policies/{summary['doc_id']}/ask", json={"question": "What is the deductible?"}).json()
    assert ask["status"] == "insufficient_evidence" and ask["evidence"] == []


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
def test_pgvector_restart_reuses_stored_vectors(tmp_path):
    pdf = make_text_pdf([["9. AMBULANCE", "Road ambulance charges are paid up to Rs 2,000 per admission."]])
    url = os.environ["TEST_DATABASE_URL"]
    first = make_client(tmp_path, database_url=url)
    assert first.get("/health").json()["retrieval"].endswith("@ pgvector")
    doc_id = first.post("/policies", files={"file": ("amb.pdf", pdf, "application/pdf")}).json()["doc_id"]

    counting = CountingEmbedder(HashingEmbedder())
    restarted = make_client(tmp_path, embedder=counting, database_url=url)
    hits = restarted.post(f"/policies/{doc_id}/search", json={"query": "ambulance charges"}).json()
    assert counting.documents_embedded == 0
    assert hits[0]["section"] == "9. AMBULANCE" and hits[0]["page"] == 1


def test_built_frontend_is_served_at_ui(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>ui</title>")
    client = make_client(tmp_path, frontend_dist=dist)
    assert client.get("/ui/").status_code == 200 and "<title>ui</title>" in client.get("/ui/").text
    assert client.get("/", follow_redirects=False).headers["location"] == "/ui/"
    assert client.get("/health").status_code == 200  # API routes unaffected


def test_frontend_not_mounted_without_a_build(tmp_path):
    client = make_client(tmp_path, frontend_dist=tmp_path / "missing")
    assert client.get("/ui/").status_code == 404
