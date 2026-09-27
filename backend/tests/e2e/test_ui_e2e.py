"""Browser test of the demo UI against a running server (skipped unless E2E_BASE_URL is set).

    cd frontend && npm run build
    cd backend && EMBEDDING_PROVIDER=hashing uvicorn app.main:create_app --factory --port 8000 &
    E2E_BASE_URL=http://localhost:8000 pytest tests/e2e -q

Optional: E2E_CHROMIUM_PATH=/path/to/chrome, E2E_SCREENSHOT_DIR=/some/dir.
The server must run without an ANTHROPIC_API_KEY (answers are then evidence-only).
"""

import os
from pathlib import Path

import pytest

BASE = os.environ.get("E2E_BASE_URL")
pytestmark = pytest.mark.skipif(not BASE, reason="E2E_BASE_URL not set")


@pytest.fixture(scope="module")
def page():
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("E2E_CHROMIUM_PATH") or None)
        page = browser.new_page(viewport={"width": 1100, "height": 1400})
        page.set_default_timeout(15000)
        yield page
        browser.close()


def shot(page, name: str) -> None:
    out = os.environ.get("E2E_SCREENSHOT_DIR")
    if out:
        page.screenshot(path=str(Path(out) / f"{name}.png"), full_page=True)


def test_demo_workflow(page, tmp_path):
    from evaluation.dataset import load_dataset

    pdf = tmp_path / "fictional_health_shield_policy.pdf"
    pdf.write_bytes(load_dataset().policy_pdf())

    page.goto(f"{BASE}/ui/")
    page.get_by_text("LLM: extractive").wait_for()

    # 1. Upload + indexing status
    page.get_by_label("Policy PDF (text-based)").set_input_files(str(pdf))
    page.get_by_role("button", name="Upload and index").click()
    status = page.get_by_test_id("policy-status")
    status.get_by_text("8 pages").wait_for()
    assert "16 chunks" in status.inner_text()

    # 2-3. Answerable question → evidence with page/section citation
    page.get_by_label("Question").fill("What is the deductible for each policy year?")
    page.get_by_role("button", name="Ask").click()
    page.get_by_test_id("answer-status").get_by_text("Evidence only").wait_for()
    citations = page.get_by_test_id("citations").inner_text()
    assert "Page 4 · 8. DEDUCTIBLE" in citations and "Rs 10,000" in citations
    shot(page, "1-evidence")

    # Unanswerable question → insufficient evidence, no answer text
    page.get_by_label("Question").fill("Is there cover for vaccinations?")
    page.get_by_role("button", name="Ask").click()
    page.get_by_test_id("answer-status").get_by_text("Insufficient evidence").wait_for()
    assert page.get_by_test_id("citations").count() == 0

    # 4-5. Treatment → synthetic estimate
    page.locator("select").filter(has_text="Cataract surgery").select_option(label="Cataract surgery (one eye)")
    page.get_by_role("button", name="Estimate cost").click()
    estimate = page.get_by_test_id("estimate")
    estimate.get_by_text("SYNTHETIC DEMO DATA").wait_for()
    assert "₹40,000" in estimate.inner_text()  # midpoint of the synthetic 25,000–55,000 range

    # 6. Coverage breakdown (deterministic): 40,000 → sub-limit 40,000 → −10,000 deductible → −10% co-pay
    page.get_by_label("Sum insured (₹)").fill("500000")
    page.get_by_label("Deductible (₹ per year)").fill("10000")
    page.get_by_label("Co-payment (%)").fill("10")
    page.get_by_label("Sub-limit for cataract (₹)").fill("40000")
    page.get_by_role("button", name="Calculate coverage").click()
    page.get_by_test_id("coverage").wait_for()
    assert page.get_by_test_id("covered").inner_text() == "₹27,000"
    assert page.get_by_test_id("oop").inner_text() == "₹13,000"
    rules = page.get_by_test_id("coverage").locator("tbody tr td:nth-child(2)").all_inner_texts()
    assert rules == ["sub_limit", "deductible", "copay", "coverage_limit"]
    shot(page, "2-coverage")
