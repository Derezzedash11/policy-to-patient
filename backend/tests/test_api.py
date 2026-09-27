import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from conftest import StubLLM


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(Settings(data_dir=tmp_path, anthropic_api_key=None), llm=None))


def upload(client, pdf):
    return client.post("/policies", files={"file": ("fictional_policy.pdf", pdf, "application/pdf")})


def test_health_reports_extractive_mode(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_mode"] == "extractive"


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
    client = TestClient(create_app(Settings(data_dir=tmp_path), llm=llm))
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


def test_policies_persist_across_app_restart(tmp_path, policy_pdf):
    first = TestClient(create_app(Settings(data_dir=tmp_path), llm=None))
    doc_id = upload(first, policy_pdf).json()["doc_id"]
    second = TestClient(create_app(Settings(data_dir=tmp_path), llm=None))
    r = second.post(f"/policies/{doc_id}/search", json={"query": "sum insured"})
    assert r.status_code == 200 and r.json()


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
