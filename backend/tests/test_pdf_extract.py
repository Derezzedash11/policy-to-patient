import pytest

from app.ingestion.pdf_extract import PDFExtractionError, extract_pages
from pdf_factory import make_text_pdf


def test_extracts_each_page_with_one_based_numbers(policy_pages):
    assert [p.page for p in policy_pages] == [1, 2, 3, 4]
    assert "FICTIONAL TEST POLICY" in policy_pages[0].text
    assert "Rs 5,000 per day" in policy_pages[1].text
    assert "deductible of Rs 10,000" in policy_pages[3].text


def test_page_without_text_is_flagged_not_invented(policy_pages):
    empty = policy_pages[2]
    assert empty.empty_text is True
    assert empty.text == ""


def test_text_stays_on_its_own_page(policy_pages):
    assert "co-payment" not in policy_pages[1].text
    assert "co-payment" in policy_pages[3].text


def test_rejects_non_pdf():
    with pytest.raises(PDFExtractionError):
        extract_pages(b"hello, not a pdf")


def test_rejects_corrupt_pdf():
    with pytest.raises(PDFExtractionError):
        extract_pages(b"%PDF-1.4\n garbage without objects")


def test_escaped_characters_survive():
    pages = extract_pages(make_text_pdf([["Limit (per day) applies"]]))
    assert pages[0].text == "Limit (per day) applies"
