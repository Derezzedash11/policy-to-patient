import pytest

from app.ingestion.chunker import chunk_pages, is_heading
from app.models import PageText


def test_every_chunk_has_source_metadata(policy_chunks, policy_pages):
    assert policy_chunks
    for c in policy_chunks:
        assert c.doc_id == "testdoc"
        assert c.doc_name == "fictional_policy.pdf"
        assert c.page >= 1
        page_text = policy_pages[c.page - 1].text
        # The chunk text is an exact slice of its page: nothing invented or merged across pages.
        assert page_text[c.char_start : c.char_end] == c.text


def test_empty_page_produces_no_chunks(policy_chunks):
    assert all(c.page != 3 for c in policy_chunks)


def test_sections_detected(policy_chunks):
    by_text = {c.section: c for c in policy_chunks}
    assert "2. COVERAGE AND LIMITS" in by_text
    assert "5. EXCLUSIONS" in by_text
    deductible = next(c for c in policy_chunks if "deductible of Rs 10,000" in c.text)
    assert deductible.section == "3. DEDUCTIBLE AND CO-PAYMENT"
    assert deductible.page == 4


def test_section_carries_over_page_break():
    pages = [
        PageText(page=1, text="7. CLAIMS\nNotify us within 48 hours.", empty_text=False),
        PageText(page=2, text="Submit bills within 30 days of discharge.", empty_text=False),
    ]
    chunks = chunk_pages(pages, "d", "x.pdf")
    assert chunks[-1].page == 2
    assert chunks[-1].section == "7. CLAIMS"


def test_no_heading_means_no_section():
    pages = [PageText(page=1, text="plain sentence about hospital cover.", empty_text=False)]
    assert chunk_pages(pages, "d", "x.pdf")[0].section is None


def test_long_text_split_with_overlap_and_offsets():
    words = [f"word{i}" for i in range(400)]
    text = " ".join(words)
    pages = [PageText(page=5, text=text, empty_text=False)]
    chunks = chunk_pages(pages, "d", "x.pdf", max_chars=300, overlap=60)
    assert len(chunks) > 5
    for c in chunks:
        assert len(c.text) <= 300
        assert text[c.char_start : c.char_end] == c.text
        assert not c.text.startswith("ord")  # windows start on word boundaries
    # Consecutive chunks overlap and together cover the whole page.
    for a, b in zip(chunks, chunks[1:]):
        assert b.char_start < a.char_end
    assert chunks[0].char_start == 0 and chunks[-1].char_end == len(text)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_pages([], "d", "x.pdf", max_chars=100, overlap=100)


@pytest.mark.parametrize(
    "line,expected",
    [
        ("4. EXCLUSIONS", True),
        ("4.2 Room Rent Limits", True),
        ("SECTION C - BENEFITS", True),
        ("Section 5 Claims", True),
        ("4.2 The deductible is Rs 10,000 for each policy year.", False),
        ("The insured person must pay the deductible.", False),
        ("", False),
    ],
)
def test_heading_heuristic(line, expected):
    assert is_heading(line) is expected
