"""Deterministic checks that a generated answer is supported by the passages it cites.

These checks cannot judge meaning. They catch the failures that matter most for insurance
answers: citing passages that were never supplied, answering without citations, and quoting
amounts, percentages or periods that do not appear in the cited policy text.
"""

from __future__ import annotations

import re

from app.retrieval.base import ScoredChunk

_CITATION_GROUP = re.compile(r"\[\s*(C\d+(?:\s*[,;]\s*C\d+)*)\s*\]")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "ninety": "90",
}


def passage_label(index: int) -> str:
    return f"C{index + 1}"


def extract_citation_labels(text: str) -> list[str]:
    """Labels cited in the text, in first-appearance order (e.g. ['C2', 'C1'])."""
    labels: list[str] = []
    for group in _CITATION_GROUP.findall(text):
        for label in re.split(r"\s*[,;]\s*", group):
            if label not in labels:
                labels.append(label)
    return labels


def validate_citations(answer: str, evidence: list[ScoredChunk]) -> tuple[list[str], list[str]]:
    """Split cited labels into (valid, invalid) against the passages actually provided."""
    allowed = {passage_label(i) for i in range(len(evidence))}
    cited = extract_citation_labels(answer)
    return [c for c in cited if c in allowed], [c for c in cited if c not in allowed]


def numbers_in(text: str) -> set[str]:
    """Numbers in text, normalised: '5,00,000' → '500000', 'twelve' → '12'."""
    text = _CITATION_GROUP.sub(" ", text)
    found = {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text)}
    words = re.findall(r"[a-z]+", text.lower())
    found |= {_NUMBER_WORDS[w] for w in words if w in _NUMBER_WORDS}
    return found


def unsupported_figures(answer: str, evidence: list[ScoredChunk]) -> list[str]:
    """Problems where a sentence states a figure that its cited passages do not contain."""
    by_label = {passage_label(i): hit.chunk.text for i, hit in enumerate(evidence)}
    problems: list[str] = []
    for sentence in _SENTENCE_END.split(answer.strip()):
        figures = numbers_in(sentence)
        if not figures:
            continue
        cited = [label for label in extract_citation_labels(sentence) if label in by_label]
        if not cited:
            problems.append(f"States {', '.join(sorted(figures))} without citing a passage: {sentence!r}")
            continue
        supported = set().union(*(numbers_in(by_label[label]) for label in cited))
        missing = sorted(figures - supported)
        if missing:
            problems.append(
                f"{', '.join(missing)} not found in cited passage(s) {', '.join(cited)}: {sentence!r}"
            )
    return problems


def verify_answer(answer: str, evidence: list[ScoredChunk]) -> tuple[list[str], list[str]]:
    """Return (valid cited labels, problems). No problems means the answer passed every check."""
    valid, invalid = validate_citations(answer, evidence)
    problems: list[str] = []
    if invalid:
        problems.append(f"Cites passages that were not provided: {', '.join(invalid)}")
    if not valid:
        problems.append("Does not cite any provided passage")
    problems += unsupported_figures(answer, evidence)
    return valid, problems
