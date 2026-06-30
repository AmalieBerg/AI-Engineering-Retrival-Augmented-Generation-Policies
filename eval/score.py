"""Pure scoring functions for the evaluation harness.

Each function takes structured inputs and returns structured outputs, with no
external dependencies. This makes scoring deterministic, testable, and
independent of which LLM produced the answers.

Metrics implemented:
  - citation_metrics: precision/recall/F1 against gold_doc_ids
  - partial_match: fraction of must_contain substrings present
  - exact_match: 1 if all must_contain are present, else 0
  - refusal_check: did the model correctly refuse a should_refuse question?
  - latency_percentiles: p50, p95, mean, min, max
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class CitationMetrics:
    """Citation precision/recall/F1 for a single answer."""
    precision: float
    recall: float
    f1: float
    cited: List[str]
    expected: List[str]
    true_positives: List[str]
    false_positives: List[str]
    false_negatives: List[str]


def citation_metrics(cited_doc_ids: Sequence[str], gold_doc_ids: Sequence[str]) -> CitationMetrics:
    """Compute precision/recall/F1 between cited and gold doc IDs.

    Both inputs are treated as sets (duplicates collapsed). A question with no
    gold citations (e.g., a refusal question) returns 1.0 for precision and
    recall iff cited is also empty -- citing anything when nothing is expected
    is incorrect.
    """
    cited_set = set(cited_doc_ids)
    gold_set = set(gold_doc_ids)

    true_positives = cited_set & gold_set
    false_positives = cited_set - gold_set
    false_negatives = gold_set - cited_set

    # Edge cases
    if not gold_set:
        # No expected citations — perfect score iff none were given
        precision = 1.0 if not cited_set else 0.0
        recall = 1.0
    else:
        precision = (len(true_positives) / len(cited_set)) if cited_set else 0.0
        recall = len(true_positives) / len(gold_set)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return CitationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        cited=sorted(cited_set),
        expected=sorted(gold_set),
        true_positives=sorted(true_positives),
        false_positives=sorted(false_positives),
        false_negatives=sorted(false_negatives),
    )


@dataclass(frozen=True)
class MatchScores:
    """Substring-based match scoring."""
    exact_match: int  # 1 or 0
    partial_match: float  # [0, 1]
    matched: List[str]
    missed: List[str]


def match_scores(answer: str, must_contain: Sequence[str]) -> MatchScores:
    """Score an answer against a list of required substrings (case-insensitive).

    For refusal questions, must_contain typically holds the canonical refusal
    phrase. For substantive questions, it holds the distinctive numbers/phrases
    that any correct answer should mention.

    Empty must_contain returns full credit (no requirements to violate).
    """
    if not must_contain:
        return MatchScores(exact_match=1, partial_match=1.0, matched=[], missed=[])

    answer_lower = answer.lower()
    matched: List[str] = []
    missed: List[str] = []
    for needle in must_contain:
        if needle.lower() in answer_lower:
            matched.append(needle)
        else:
            missed.append(needle)

    n = len(must_contain)
    partial = len(matched) / n if n else 1.0
    exact = 1 if not missed else 0
    return MatchScores(
        exact_match=exact,
        partial_match=partial,
        matched=matched,
        missed=missed,
    )


def refusal_check(answer: str, should_refuse: bool, refusal_phrase: str) -> bool:
    """Return True iff the model's refusal behavior matched the expectation.

    - If should_refuse=True: the answer must contain the canonical refusal phrase
    - If should_refuse=False: the answer must NOT be a refusal
    """
    normalized_answer = answer.strip().rstrip(".").lower()
    normalized_target = refusal_phrase.strip().rstrip(".").lower()
    is_refusal = normalized_target in normalized_answer

    if should_refuse:
        return is_refusal
    else:
        return not is_refusal


@dataclass(frozen=True)
class LatencyStats:
    """Latency aggregates over a set of question runs."""
    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float
    n: int


def latency_percentiles(latencies_ms: Sequence[float]) -> LatencyStats:
    """Compute p50/p95/p99/mean/min/max latency stats.

    Uses linear interpolation between data points (the standard "nearest-rank
    via linear" method). Returns zeros for an empty list.
    """
    if not latencies_ms:
        return LatencyStats(p50=0.0, p95=0.0, p99=0.0, mean=0.0, min=0.0, max=0.0, n=0)

    sorted_latencies = sorted(latencies_ms)
    n = len(sorted_latencies)

    def percentile(p: float) -> float:
        if n == 1:
            return sorted_latencies[0]
        # Linear interpolation between adjacent ranks
        rank = (p / 100.0) * (n - 1)
        lo_idx = int(rank)
        hi_idx = min(lo_idx + 1, n - 1)
        frac = rank - lo_idx
        return sorted_latencies[lo_idx] + frac * (sorted_latencies[hi_idx] - sorted_latencies[lo_idx])

    return LatencyStats(
        p50=percentile(50),
        p95=percentile(95),
        p99=percentile(99),
        mean=sum(sorted_latencies) / n,
        min=sorted_latencies[0],
        max=sorted_latencies[-1],
        n=n,
    )


@dataclass(frozen=True)
class AggregatedScores:
    """Aggregated metrics across all eval questions."""
    n_questions: int
    n_substantive: int  # Non-refusal questions
    n_refusal: int  # Refusal questions
    # Per-metric aggregates (mean across questions where applicable)
    citation_precision_mean: float
    citation_recall_mean: float
    citation_f1_mean: float
    partial_match_mean: float
    exact_match_rate: float
    refusal_rate: float  # Among refusal questions, % correctly refused
    groundedness_rate: float  # % of all answers judged grounded
    # Latency
    latency: LatencyStats


def aggregate(per_question_scores: List[dict]) -> AggregatedScores:
    """Aggregate a list of per-question score records into summary statistics.

    Each input record is expected to contain keys:
      - should_refuse (bool)
      - citation_metrics (CitationMetrics)
      - match (MatchScores)
      - refusal_correct (bool, only for refusal questions)
      - grounded (bool or None — None when judge wasn't run)
      - latency_ms (float)
    """
    n = len(per_question_scores)
    if n == 0:
        return AggregatedScores(
            n_questions=0, n_substantive=0, n_refusal=0,
            citation_precision_mean=0.0, citation_recall_mean=0.0,
            citation_f1_mean=0.0, partial_match_mean=0.0,
            exact_match_rate=0.0, refusal_rate=0.0, groundedness_rate=0.0,
            latency=latency_percentiles([]),
        )

    substantive = [r for r in per_question_scores if not r["should_refuse"]]
    refusals = [r for r in per_question_scores if r["should_refuse"]]
    grounded_judgments = [r["grounded"] for r in per_question_scores if r.get("grounded") is not None]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return AggregatedScores(
        n_questions=n,
        n_substantive=len(substantive),
        n_refusal=len(refusals),
        citation_precision_mean=mean([r["citation_metrics"].precision for r in substantive]),
        citation_recall_mean=mean([r["citation_metrics"].recall for r in substantive]),
        citation_f1_mean=mean([r["citation_metrics"].f1 for r in substantive]),
        partial_match_mean=mean([r["match"].partial_match for r in substantive]),
        exact_match_rate=mean([float(r["match"].exact_match) for r in substantive]),
        refusal_rate=mean([float(r["refusal_correct"]) for r in refusals]) if refusals else 1.0,
        groundedness_rate=mean([float(g) for g in grounded_judgments]),
        latency=latency_percentiles([r["latency_ms"] for r in per_question_scores]),
    )
