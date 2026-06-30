"""Tests for rag.prompt — citation parsing, refusal detection, prompt formatting."""
from __future__ import annotations

from rag.prompt import (
    REFUSAL_PHRASE,
    build_system_prompt,
    build_user_prompt,
    format_context_block,
    is_refusal,
    parse_citations,
)
from rag.types import RetrievedChunk


# --- parse_citations -------------------------------------------------------

def test_parse_single_citation():
    text = "PTO accrues at 15 days per year [POL-HR-001]."
    assert parse_citations(text) == ["POL-HR-001"]


def test_parse_multiple_unique_citations():
    text = "Source code is not accessible from personal devices [POL-IT-003][POL-SEC-001]."
    assert parse_citations(text) == ["POL-IT-003", "POL-SEC-001"]


def test_parse_dedupes_citations():
    text = "[POL-HR-001] says PTO accrues. [POL-HR-001] also defines carryover."
    assert parse_citations(text) == ["POL-HR-001"]


def test_parse_preserves_order_of_first_appearance():
    text = "First [POL-HR-002], then [POL-HR-001], then [POL-HR-002] again."
    assert parse_citations(text) == ["POL-HR-002", "POL-HR-001"]


def test_parse_returns_empty_for_no_citations():
    assert parse_citations("Just a plain answer with no sources.") == []


def test_parse_ignores_malformed_brackets():
    # POL-FOO-1 (too short) and pol-hr-001 (lowercase) and [POL-HR-001 missing close]
    text = "[POL-FOO-1] [pol-hr-001] [POL-HR-001 something] but [POL-LEG-001] is good"
    # Only POL-LEG-001 matches the strict pattern
    assert parse_citations(text) == ["POL-LEG-001"]


def test_parse_normalizes_case():
    # The regex is case-sensitive on POL-XX-NNN; we only normalize matches.
    text = "[POL-HR-001] and [POL-HR-002]"
    assert parse_citations(text) == ["POL-HR-001", "POL-HR-002"]


# --- is_refusal ------------------------------------------------------------

def test_is_refusal_exact_match():
    assert is_refusal(REFUSAL_PHRASE)


def test_is_refusal_with_trailing_whitespace():
    assert is_refusal(REFUSAL_PHRASE + "  \n")


def test_is_refusal_case_insensitive():
    assert is_refusal(REFUSAL_PHRASE.upper())


def test_is_refusal_false_for_answer_with_content():
    assert not is_refusal("Employees accrue 15 days of PTO per year [POL-HR-001].")


def test_is_refusal_false_for_empty():
    assert not is_refusal("")


def test_is_refusal_true_when_embedded_in_longer_text():
    # The model might prefix the refusal with a greeting; we should still detect it.
    text = REFUSAL_PHRASE + " You may want to contact HR."
    assert is_refusal(text)


# --- format_context_block --------------------------------------------------

def test_format_context_block_with_full_metadata():
    chunk = RetrievedChunk(
        content="PTO accrues at 15 days for 0-2 years of service.",
        doc_id="POL-HR-001",
        doc_title="PTO and Vacation Policy",
        source_path="md/pto-vacation-policy.md",
        section="3. PTO Accrual",
        chunk_index=2,
        score=0.1,
    )
    block = format_context_block(chunk)
    assert "[POL-HR-001 - PTO and Vacation Policy § 3. PTO Accrual]" in block
    assert "PTO accrues at 15 days" in block


def test_format_context_block_without_section():
    chunk = RetrievedChunk(
        content="Some plain content.",
        doc_id="POL-FIN-001",
        doc_title="Expense Policy",
        source_path="md/expense.md",
        section=None,
        chunk_index=0,
        score=0.0,
    )
    block = format_context_block(chunk)
    assert "[POL-FIN-001 - Expense Policy]" in block
    assert "Some plain content." in block


# --- build_user_prompt -----------------------------------------------------

def test_build_user_prompt_with_chunks():
    chunks = [
        RetrievedChunk(content="PTO is 15 days.", doc_id="POL-HR-001",
                       doc_title="PTO", source_path="x", section="3", chunk_index=0, score=0.1),
    ]
    prompt = build_user_prompt("How much PTO do I get?", chunks)
    assert "How much PTO do I get?" in prompt
    assert "POL-HR-001" in prompt
    assert "PTO is 15 days." in prompt


def test_build_user_prompt_with_empty_chunks():
    """Empty context should be passed through honestly so the model refuses."""
    prompt = build_user_prompt("Random unrelated question", [])
    assert "Random unrelated question" in prompt
    assert "no relevant policy excerpts" in prompt.lower()


# --- build_system_prompt ---------------------------------------------------

def test_system_prompt_contains_refusal_phrase():
    """The exact refusal phrase must appear in the system prompt so the model can copy it."""
    sp = build_system_prompt()
    assert REFUSAL_PHRASE in sp


def test_system_prompt_mentions_citation_format():
    sp = build_system_prompt()
    assert "POL-" in sp  # Format example
    assert "cite" in sp.lower()
