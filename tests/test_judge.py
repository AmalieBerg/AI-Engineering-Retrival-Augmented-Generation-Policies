"""Tests for eval.judge — verdict parsing must be robust to LLM output variation."""
from __future__ import annotations

import pytest

from eval.judge import (
    JudgeVerdict,
    format_context_for_judge,
    parse_verdict,
)


# --- parse_verdict ---------------------------------------------------------

def test_parse_clean_json_grounded_true():
    v = parse_verdict('{"grounded": true, "rationale": "All claims supported."}')
    assert v.grounded is True
    assert v.rationale == "All claims supported."


def test_parse_clean_json_grounded_false():
    v = parse_verdict('{"grounded": false, "rationale": "Number 20 is not in context."}')
    assert v.grounded is False
    assert "Number 20" in v.rationale


def test_parse_json_in_code_fence():
    raw = '```json\n{"grounded": true, "rationale": "OK"}\n```'
    v = parse_verdict(raw)
    assert v.grounded is True


def test_parse_json_in_plain_code_fence():
    raw = '```\n{"grounded": false, "rationale": "Fabricated date"}\n```'
    v = parse_verdict(raw)
    assert v.grounded is False


def test_parse_with_leading_prose():
    raw = 'Here is my verdict:\n\n{"grounded": true, "rationale": "Good"}'
    v = parse_verdict(raw)
    assert v.grounded is True


def test_parse_with_trailing_prose():
    raw = '{"grounded": false, "rationale": "Missing source"} Hope this helps!'
    v = parse_verdict(raw)
    assert v.grounded is False


def test_parse_python_boolean_literals():
    """Some models emit True/False instead of true/false."""
    raw = '{"grounded": True, "rationale": "yep"}'
    v = parse_verdict(raw)
    assert v.grounded is True


def test_parse_string_true():
    raw = '{"grounded": "true", "rationale": "Stringified"}'
    v = parse_verdict(raw)
    assert v.grounded is True


def test_parse_empty_output():
    v = parse_verdict("")
    assert v.grounded is None
    assert "Empty" in v.rationale


def test_parse_garbage_output():
    v = parse_verdict("This is not JSON at all, just some prose.")
    assert v.grounded is None


def test_parse_missing_grounded_field():
    v = parse_verdict('{"rationale": "no verdict field"}')
    assert v.grounded is None


def test_parse_preserves_raw_output():
    raw = '{"grounded": true, "rationale": "ok"}'
    v = parse_verdict(raw)
    assert v.raw == raw


# --- format_context_for_judge ----------------------------------------------

def test_format_context_empty_chunks():
    assert format_context_for_judge([]) == "(no chunks retrieved)"


def test_format_context_single_chunk():
    chunks = [{
        "doc_id": "POL-HR-001",
        "doc_title": "PTO Policy",
        "section": "3. Accrual",
        "content": "PTO accrues at 15 days per year for the first 2 years.",
    }]
    out = format_context_for_judge(chunks)
    assert "POL-HR-001" in out
    assert "PTO Policy" in out
    assert "3. Accrual" in out
    assert "15 days" in out


def test_format_context_multiple_chunks_separated():
    chunks = [
        {"doc_id": "POL-A-001", "doc_title": "A", "section": "1", "content": "Content A"},
        {"doc_id": "POL-B-002", "doc_title": "B", "section": "2", "content": "Content B"},
    ]
    out = format_context_for_judge(chunks)
    assert "POL-A-001" in out
    assert "POL-B-002" in out
    # Blocks should be separated by a blank line
    assert "Content A" in out and "Content B" in out
    # Order preserved
    assert out.index("POL-A-001") < out.index("POL-B-002")


def test_format_context_missing_section_handled():
    chunks = [{
        "doc_id": "POL-HR-001",
        "doc_title": "PTO Policy",
        "content": "Some content.",
    }]
    out = format_context_for_judge(chunks)
    assert "POL-HR-001" in out
    assert "Some content." in out
