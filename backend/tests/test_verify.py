import pytest

from app.models import Chunk
from app.qa.rewrite import rewrite_query
from app.qa.verify import numbers_in, unsupported_figures, verify_answer
from app.retrieval.base import ScoredChunk
from conftest import ScriptedLLM


def _evidence(*texts: str) -> list[ScoredChunk]:
    return [
        ScoredChunk(Chunk(chunk_id=f"d:p1:c{i}", doc_id="d", doc_name="x.pdf", page=1, section=None,
                          text=t, char_start=0, char_end=len(t)), 0.9)
        for i, t in enumerate(texts)
    ]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Rs 5,00,000 per year", {"500000"}),
        ("10 percent and 24 months.", {"10", "24"}),
        ("twelve months, two deliveries", {"12", "2"}),
        ("See [C3] and [C1, C2].", set()),  # citation labels are not figures
        ("1.5 lakh", {"1.5"}),
        ("no figures here", set()),
    ],
)
def test_numbers_in(text, expected):
    assert numbers_in(text) == expected


def test_figures_must_come_from_the_cited_passage_not_another_one():
    evidence = _evidence("Deductible is Rs 10,000.", "Room rent up to Rs 5,000 per day.")
    problems = unsupported_figures("Room rent is capped at Rs 10,000 per day [C2].", evidence)
    assert problems and "10000 not found in cited passage(s) C2" in problems[0]
    assert unsupported_figures("Room rent is capped at Rs 5,000 per day [C2].", evidence) == []


def test_sentence_citing_several_passages_may_use_figures_from_any_of_them():
    evidence = _evidence("Deductible is Rs 10,000.", "Co-payment of 10 percent.")
    assert unsupported_figures("You pay Rs 10,000 and then 10% [C1][C2].", evidence) == []


def test_answer_without_figures_only_needs_valid_citations():
    evidence = _evidence("Dental treatment is excluded.")
    assert verify_answer("Dental treatment is excluded [C1].", evidence) == (["C1"], [])
    _, problems = verify_answer("Dental treatment is excluded.", evidence)
    assert problems == ["Does not cite any provided passage"]


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("maternity waiting period", "maternity waiting period"),
        ('Query: "co-payment percentage"\nextra line', "co-payment percentage"),
        ("", None),
        ("   \n  ", None),
        ("What is covered?", None),  # identical to an already-tried query (case-insensitive)
    ],
)
def test_rewrite_query_sanitises_output(reply, expected):
    assert rewrite_query(ScriptedLLM(reply), "what is COVERED?", ["What is covered?"]) == expected


def test_rewrite_query_is_length_capped():
    assert len(rewrite_query(ScriptedLLM("x" * 500), "q", ["q"])) == 200
