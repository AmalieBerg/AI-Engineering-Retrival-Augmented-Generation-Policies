"""Tests for rag.guardrails."""
from __future__ import annotations

from rag.guardrails import (
    MAX_QUESTION_LENGTH,
    MIN_QUESTION_LENGTH,
    filter_invalid_citations,
    validate_answer,
    validate_question,
)
from rag.prompt import REFUSAL_PHRASE


# --- validate_question -----------------------------------------------------

def test_validate_question_accepts_normal_question():
    result = validate_question("How much PTO do I get after 4 years?")
    assert result.ok
    assert result.reason is None


def test_validate_question_rejects_none():
    result = validate_question(None)  # type: ignore[arg-type]
    assert not result.ok


def test_validate_question_rejects_non_string():
    result = validate_question(123)  # type: ignore[arg-type]
    assert not result.ok


def test_validate_question_rejects_too_short():
    result = validate_question("hi")
    assert not result.ok


def test_validate_question_accepts_min_length_question():
    q = "x" * MIN_QUESTION_LENGTH
    assert validate_question(q).ok


def test_validate_question_rejects_too_long():
    q = "x" * (MAX_QUESTION_LENGTH + 1)
    assert not validate_question(q).ok


def test_validate_question_rejects_whitespace_only():
    assert not validate_question("   \n   ").ok


# --- validate_answer -------------------------------------------------------

def test_validate_answer_passes_grounded_answer_with_citations():
    answer = "You accrue 15 days of PTO per year for the first 2 years [POL-HR-001]."
    result = validate_answer(answer, had_retrieved_chunks=True)
    assert result.ok


def test_validate_answer_passes_canonical_refusal():
    result = validate_answer(REFUSAL_PHRASE, had_retrieved_chunks=True)
    assert result.ok


def test_validate_answer_passes_refusal_even_with_no_chunks():
    """Refusal should pass regardless of whether chunks were retrieved."""
    result = validate_answer(REFUSAL_PHRASE, had_retrieved_chunks=False)
    assert result.ok


def test_validate_answer_replaces_empty_answer():
    result = validate_answer("", had_retrieved_chunks=True)
    assert not result.ok
    assert result.replacement == REFUSAL_PHRASE


def test_validate_answer_replaces_substantive_answer_without_citations():
    """A claim-laden answer with no citations is a hallucination signal."""
    answer = "The CEO of Northwind is John Smith and he was hired in 2019."
    result = validate_answer(answer, had_retrieved_chunks=True)
    assert not result.ok
    assert result.replacement == REFUSAL_PHRASE


def test_validate_answer_replaces_substantive_answer_without_chunks():
    """If we provided no context and the model answered anyway, it's hallucinating."""
    answer = "The answer is 42 [POL-HR-001]."  # Even with a citation!
    result = validate_answer(answer, had_retrieved_chunks=False)
    assert not result.ok


def test_validate_answer_allows_short_response_without_citations():
    """'Yes.' is too short to need a citation."""
    result = validate_answer("Yes.", had_retrieved_chunks=True)
    assert result.ok


def test_validate_answer_requires_citation_for_short_fact_bearing_claim():
    """'Up to 15 days.' is short but has a number -- needs a citation."""
    result = validate_answer("Up to 15 days.", had_retrieved_chunks=True)
    assert not result.ok


def test_validate_answer_truncates_excessively_long_answer():
    answer = ("Word " * 2000) + "[POL-HR-001]"  # > 4000 chars
    result = validate_answer(answer, had_retrieved_chunks=True)
    assert not result.ok
    assert result.replacement is not None
    assert len(result.replacement) <= 4100  # Truncated + ellipsis


# --- filter_invalid_citations ----------------------------------------------

def test_filter_invalid_citations_keeps_valid_drops_invalid():
    cited = ["POL-HR-001", "POL-FAKE-999", "POL-FIN-001"]
    valid_set = {"POL-HR-001", "POL-FIN-001", "POL-HR-002"}
    valid, invalid = filter_invalid_citations(cited, valid_set)
    assert valid == ["POL-HR-001", "POL-FIN-001"]
    assert invalid == ["POL-FAKE-999"]


def test_filter_invalid_citations_empty_input():
    valid, invalid = filter_invalid_citations([], {"POL-HR-001"})
    assert valid == []
    assert invalid == []


def test_filter_invalid_citations_preserves_order():
    cited = ["POL-C", "POL-A", "POL-B"]
    valid_set = {"POL-A", "POL-B", "POL-C"}
    valid, invalid = filter_invalid_citations(cited, valid_set)
    assert valid == ["POL-C", "POL-A", "POL-B"]
