"""Tests for eval.score - all pure functions, no external dependencies.

These tests anchor the entire evaluation methodology. A bug here would
silently corrupt every metric reported in design-and-evaluation.md.
"""
from __future__ import annotations

import pytest

from eval.score import (
    aggregate,
    citation_metrics,
    latency_percentiles,
    match_scores,
    refusal_check,
)


# --- citation_metrics ------------------------------------------------------

def test_citation_perfect_match():
    m = citation_metrics(["POL-HR-001"], ["POL-HR-001"])
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_citation_perfect_match_multiple():
    m = citation_metrics(["POL-HR-001", "POL-FIN-001"], ["POL-FIN-001", "POL-HR-001"])
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_citation_partial_recall():
    # Expected 2 docs, model only cited 1 of them
    m = citation_metrics(["POL-HR-001"], ["POL-HR-001", "POL-FIN-001"])
    assert m.precision == 1.0
    assert m.recall == 0.5
    assert abs(m.f1 - 2 / 3) < 1e-9


def test_citation_partial_precision():
    # Model cited 2 docs, only 1 was expected
    m = citation_metrics(["POL-HR-001", "POL-FAKE-999"], ["POL-HR-001"])
    assert m.precision == 0.5
    assert m.recall == 1.0
    assert abs(m.f1 - 2 / 3) < 1e-9


def test_citation_no_overlap():
    m = citation_metrics(["POL-WRONG-001"], ["POL-HR-001"])
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_citation_empty_cited():
    """Model cited nothing but expected something."""
    m = citation_metrics([], ["POL-HR-001"])
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_citation_refusal_question_no_cites_no_gold():
    """Refusal question: no citations expected, none given -> perfect."""
    m = citation_metrics([], [])
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_citation_refusal_question_with_invented_cite():
    """Refusal question but model cited something -> precision is 0."""
    m = citation_metrics(["POL-HR-001"], [])
    assert m.precision == 0.0
    assert m.recall == 1.0
    assert m.f1 == 0.0


def test_citation_dedupes_inputs():
    """Duplicate doc_ids in either list should be collapsed."""
    m = citation_metrics(
        ["POL-HR-001", "POL-HR-001", "POL-FIN-001"],
        ["POL-HR-001", "POL-HR-001"],
    )
    assert m.precision == 0.5  # cited set: {HR-001, FIN-001}; only HR-001 expected
    assert m.recall == 1.0


def test_citation_records_tp_fp_fn():
    m = citation_metrics(
        ["POL-HR-001", "POL-FAKE-999"],
        ["POL-HR-001", "POL-FIN-001"],
    )
    assert m.true_positives == ["POL-HR-001"]
    assert m.false_positives == ["POL-FAKE-999"]
    assert m.false_negatives == ["POL-FIN-001"]


# --- match_scores ----------------------------------------------------------

def test_match_all_present():
    answer = "You accrue 20 days of PTO per year [POL-HR-001]."
    m = match_scores(answer, ["20"])
    assert m.exact_match == 1
    assert m.partial_match == 1.0
    assert m.missed == []


def test_match_case_insensitive():
    answer = "BREAKFAST is $20"
    m = match_scores(answer, ["$20", "breakfast"])
    assert m.exact_match == 1
    assert m.partial_match == 1.0


def test_match_partial():
    answer = "PTO is 15 days per year"
    m = match_scores(answer, ["15", "carry over"])
    assert m.exact_match == 0
    assert m.partial_match == 0.5
    assert m.matched == ["15"]
    assert m.missed == ["carry over"]


def test_match_none_present():
    m = match_scores("Wrong answer", ["correct"])
    assert m.exact_match == 0
    assert m.partial_match == 0.0


def test_match_empty_requirements():
    """No requirements = full credit."""
    m = match_scores("anything", [])
    assert m.exact_match == 1
    assert m.partial_match == 1.0
    assert m.matched == []
    assert m.missed == []


def test_match_multiple_requirements_proportional():
    answer = "$20 breakfast, $30 lunch, $60 dinner"
    m = match_scores(answer, ["$20", "$30", "$60", "$110"])
    assert m.partial_match == 0.75
    assert m.exact_match == 0
    assert len(m.matched) == 3


# --- refusal_check ---------------------------------------------------------

REFUSAL = "I can only answer about our policies, and I couldn't find this in our documents."


def test_refusal_correct_refusal():
    assert refusal_check(REFUSAL, should_refuse=True, refusal_phrase=REFUSAL)


def test_refusal_correct_non_refusal():
    assert refusal_check("PTO is 15 days [POL-HR-001].", should_refuse=False, refusal_phrase=REFUSAL)


def test_refusal_missed_refusal_question():
    """Model answered when it should have refused."""
    assert not refusal_check(
        "The dress code is business casual.",
        should_refuse=True,
        refusal_phrase=REFUSAL,
    )


def test_refusal_unwanted_refusal():
    """Model refused when it should have answered."""
    assert not refusal_check(REFUSAL, should_refuse=False, refusal_phrase=REFUSAL)


def test_refusal_tolerates_whitespace_and_case():
    assert refusal_check(REFUSAL.upper() + "\n\n", should_refuse=True, refusal_phrase=REFUSAL)


# --- latency_percentiles ---------------------------------------------------

def test_latency_empty_list():
    s = latency_percentiles([])
    assert s.n == 0
    assert s.p50 == 0.0
    assert s.mean == 0.0


def test_latency_single_value():
    s = latency_percentiles([100.0])
    assert s.p50 == 100.0
    assert s.p95 == 100.0
    assert s.mean == 100.0
    assert s.min == 100.0
    assert s.max == 100.0


def test_latency_basic_stats():
    s = latency_percentiles([100, 200, 300, 400, 500])
    assert s.n == 5
    assert s.min == 100
    assert s.max == 500
    assert s.mean == 300
    assert s.p50 == 300  # median of 5 values


def test_latency_p95_with_outlier():
    """A heavy tail should show up in p95."""
    values = [100] * 19 + [10000]  # 19 fast, 1 very slow
    s = latency_percentiles(values)
    assert s.p50 == 100
    assert s.p95 > 100  # outlier pulls p95 up
    assert s.max == 10000


def test_latency_handles_unsorted_input():
    s_unsorted = latency_percentiles([500, 100, 300, 200, 400])
    s_sorted = latency_percentiles([100, 200, 300, 400, 500])
    assert s_unsorted.p50 == s_sorted.p50
    assert s_unsorted.mean == s_sorted.mean


# --- aggregate -------------------------------------------------------------

def _make_record(qid, should_refuse=False, cite_f1=1.0, partial=1.0,
                 exact=1, refusal_correct=True, grounded=True, latency_ms=500.0,
                 cite_precision=1.0, cite_recall=1.0):
    """Helper to build a per-question record for aggregate() input."""
    from eval.score import CitationMetrics, MatchScores
    return {
        "id": qid,
        "should_refuse": should_refuse,
        "citation_metrics": CitationMetrics(
            precision=cite_precision, recall=cite_recall, f1=cite_f1,
            cited=[], expected=[], true_positives=[], false_positives=[], false_negatives=[],
        ),
        "match": MatchScores(exact_match=exact, partial_match=partial, matched=[], missed=[]),
        "refusal_correct": refusal_correct,
        "grounded": grounded,
        "latency_ms": latency_ms,
    }


def test_aggregate_empty():
    a = aggregate([])
    assert a.n_questions == 0
    assert a.citation_f1_mean == 0.0


def test_aggregate_all_perfect():
    records = [_make_record(f"Q{i}") for i in range(5)]
    a = aggregate(records)
    assert a.n_questions == 5
    assert a.n_substantive == 5
    assert a.n_refusal == 0
    assert a.citation_f1_mean == 1.0
    assert a.partial_match_mean == 1.0
    assert a.exact_match_rate == 1.0
    assert a.groundedness_rate == 1.0
    assert a.refusal_rate == 1.0  # Vacuous when no refusal questions


def test_aggregate_with_refusal_questions():
    records = [
        _make_record("Q1"),  # substantive, perfect
        _make_record("Q2", should_refuse=True, cite_f1=1.0, refusal_correct=True),
        _make_record("Q3", should_refuse=True, cite_f1=1.0, refusal_correct=False),
    ]
    a = aggregate(records)
    assert a.n_substantive == 1
    assert a.n_refusal == 2
    assert a.refusal_rate == 0.5  # 1 of 2 refusals correct


def test_aggregate_excludes_refusal_from_citation_metric():
    """Refusal questions shouldn't pull down citation F1 (they have no gold cites)."""
    records = [
        _make_record("Q1", cite_f1=1.0),
        _make_record("Q2", should_refuse=True, cite_f1=0.0),
    ]
    a = aggregate(records)
    # Citation F1 averaged only over substantive questions
    assert a.citation_f1_mean == 1.0


def test_aggregate_handles_missing_groundedness():
    """If judge wasn't run, grounded=None — should not be counted in rate."""
    records = [
        _make_record("Q1", grounded=True),
        _make_record("Q2", grounded=False),
        _make_record("Q3", grounded=None),
    ]
    a = aggregate(records)
    # Only 2 grounded judgments: 1 true / 2 total = 0.5
    assert a.groundedness_rate == 0.5


def test_aggregate_latency_aggregation():
    records = [_make_record(f"Q{i}", latency_ms=lat)
               for i, lat in enumerate([100, 200, 300, 400, 500])]
    a = aggregate(records)
    assert a.latency.n == 5
    assert a.latency.mean == 300
    assert a.latency.p50 == 300
