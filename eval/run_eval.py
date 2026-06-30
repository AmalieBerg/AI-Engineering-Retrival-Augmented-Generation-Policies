"""Main evaluation harness.

Runs the configured RAG pipeline against the eval set, scores each answer
across all five required metric families, and writes both a JSON results file
and a markdown summary suitable for inclusion in design-and-evaluation.md.

Usage:
  python -m eval.run_eval
  python -m eval.run_eval --eval-set eval/eval_set.json --output eval/results/run.json
  python -m eval.run_eval --no-judge   # Skip groundedness (faster, costs nothing)

Ablation usage:
  python -m eval.run_eval --output eval/results/k3.json   # with RETRIEVAL_K=3 in env
  python -m eval.run_eval --output eval/results/k8.json   # with RETRIEVAL_K=8 in env
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from eval.judge import GroundednessJudge  # noqa: E402
from eval.score import (  # noqa: E402
    AggregatedScores,
    aggregate,
    citation_metrics,
    match_scores,
    refusal_check,
)
from rag.llm import GroqLLMClient, LLMClient  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.prompt import REFUSAL_PHRASE, parse_citations  # noqa: E402


_DEFAULT_EVAL_SET = Path(__file__).resolve().parent / "eval_set.json"
_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_evaluation(
    eval_set_path: Path,
    pipeline: Optional[RAGPipeline] = None,
    judge: Optional[GroundednessJudge] = None,
    judge_enabled: bool = True,
) -> dict:
    """Run the full evaluation and return a results dict.

    The dict contains:
      - meta: run timestamp, config used, eval set path
      - per_question: list of detailed per-question records
      - aggregates: AggregatedScores dict
    """
    eval_set = json.loads(eval_set_path.read_text(encoding="utf-8"))
    print(f"[eval] Loaded {len(eval_set)} questions from {eval_set_path}")

    pipeline = pipeline or RAGPipeline()
    if judge_enabled and judge is None:
        judge_llm: LLMClient = GroqLLMClient()
        judge = GroundednessJudge(llm=judge_llm)
    elif not judge_enabled:
        judge = None

    per_question: List[dict] = []

    for i, q in enumerate(eval_set, start=1):
        qid = q["id"]
        question = q["question"]
        gold_doc_ids = q.get("gold_doc_ids", [])
        must_contain = q.get("must_contain", [])
        should_refuse = bool(q.get("should_refuse", False))

        print(f"[eval] ({i}/{len(eval_set)}) {qid}: {question[:70]}")

        # 1) Run the pipeline (include chunks so the judge can see them)
        response = pipeline.answer(question, include_chunks_in_response=True)

        # 2) Score metrics that don't need an LLM call
        cited = parse_citations(response.answer)
        cite_scores = citation_metrics(cited, gold_doc_ids)
        m_scores = match_scores(response.answer, must_contain)
        refusal_ok = refusal_check(response.answer, should_refuse, REFUSAL_PHRASE)

        # 3) Groundedness (LLM-as-judge)
        grounded: Optional[bool] = None
        groundedness_rationale = ""
        if judge is not None:
            verdict = judge.judge(question, response.retrieved_chunks, response.answer)
            grounded = verdict.grounded
            groundedness_rationale = verdict.rationale

        record = {
            "id": qid,
            "question": question,
            "topic": q.get("topic", ""),
            "difficulty": q.get("difficulty", ""),
            "should_refuse": should_refuse,
            "answer": response.answer,
            "cited_doc_ids": cited,
            "gold_doc_ids": gold_doc_ids,
            "must_contain": must_contain,
            "citation_metrics": cite_scores,  # CitationMetrics dataclass
            "match": m_scores,                # MatchScores dataclass
            "refusal_correct": refusal_ok,
            "grounded": grounded,
            "groundedness_rationale": groundedness_rationale,
            "latency_ms": response.latency_ms,
            "guardrail_action": response.guardrail_action,
            "invalid_citations": response.invalid_citations,
            "n_chunks_retrieved": len(response.retrieved_chunks),
        }
        per_question.append(record)

        # Live progress: print a compact one-line summary
        emoji_ok = "" if cite_scores.f1 >= 0.5 or (should_refuse and refusal_ok) else ""
        cite_str = f"cite_F1={cite_scores.f1:.2f}"
        match_str = f"match={m_scores.partial_match:.2f}"
        grounded_str = f"grounded={grounded}" if grounded is not None else "grounded=skip"
        refusal_str = f"refused_correctly={refusal_ok}" if should_refuse else ""
        latency_str = f"{response.latency_ms:.0f}ms"
        print(f"        {emoji_ok} {cite_str} {match_str} {grounded_str} {refusal_str} {latency_str}")

    # 4) Aggregate
    aggregates = aggregate(per_question)

    # 5) Build the results document
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_set_path": str(eval_set_path),
        "config": {
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "retrieval_k": settings.retrieval_k,
            "fetch_k": settings.fetch_k,
            "use_mmr": settings.use_mmr,
            "use_reranker": settings.use_reranker,
        },
        "judge_enabled": judge is not None,
    }

    return {
        "meta": meta,
        "per_question": [_serialize_record(r) for r in per_question],
        "aggregates": _serialize_aggregates(aggregates),
    }


def _serialize_record(record: dict) -> dict:
    """Convert dataclass fields to dicts for JSON serialization."""
    out = dict(record)
    out["citation_metrics"] = asdict(record["citation_metrics"])
    out["match"] = asdict(record["match"])
    return out


def _serialize_aggregates(agg: AggregatedScores) -> dict:
    """Aggregates dataclass -> dict (with nested LatencyStats also flattened)."""
    out = asdict(agg)
    # asdict(asdict()) already handles nested dataclasses, but let's be explicit
    return out


# ----- Markdown summary -----

def write_markdown_summary(results: dict, output_path: Path) -> None:
    """Write a human-readable markdown summary suitable for design-and-evaluation.md."""
    agg = results["aggregates"]
    meta = results["meta"]
    lat = agg["latency"]

    lines = [
        "# Evaluation Results",
        "",
        f"**Run timestamp:** `{meta['timestamp']}`",
        f"**Eval set:** `{meta['eval_set_path']}`",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "|---------|-------|",
        f"| LLM model | `{meta['config']['llm_model']}` |",
        f"| Embedding model | `{meta['config']['embedding_model']}` |",
        f"| Chunk size | {meta['config']['chunk_size']} |",
        f"| Chunk overlap | {meta['config']['chunk_overlap']} |",
        f"| Retrieval k | {meta['config']['retrieval_k']} |",
        f"| Use MMR | {meta['config']['use_mmr']} |",
        f"| Use reranker | {meta['config']['use_reranker']} |",
        f"| Groundedness judge enabled | {meta['judge_enabled']} |",
        "",
        "## Headline Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Questions evaluated | {agg['n_questions']} (substantive: {agg['n_substantive']}, refusal: {agg['n_refusal']}) |",
        f"| **Groundedness rate** | **{agg['groundedness_rate']:.1%}** |",
        f"| **Citation F1** (mean) | **{agg['citation_f1_mean']:.3f}** |",
        f"| Citation precision (mean) | {agg['citation_precision_mean']:.3f} |",
        f"| Citation recall (mean) | {agg['citation_recall_mean']:.3f} |",
        f"| Partial-match (mean) | {agg['partial_match_mean']:.3f} |",
        f"| Exact-match rate | {agg['exact_match_rate']:.1%} |",
        f"| **Refusal rate** (should-refuse questions) | **{agg['refusal_rate']:.1%}** |",
        "",
        "## Latency",
        "",
        "| Percentile | Latency (ms) |",
        "|------------|--------------|",
        f"| p50 (median) | {lat['p50']:.0f} |",
        f"| p95 | {lat['p95']:.0f} |",
        f"| p99 | {lat['p99']:.0f} |",
        f"| mean | {lat['mean']:.0f} |",
        f"| min | {lat['min']:.0f} |",
        f"| max | {lat['max']:.0f} |",
        f"| n | {lat['n']} |",
        "",
        "## Per-Question Detail",
        "",
        "| ID | Topic | Cite F1 | Partial | Grounded | Refusal OK | Latency (ms) |",
        "|----|-------|---------|---------|----------|-----------|--------------|",
    ]

    for r in results["per_question"]:
        cite_f1 = r["citation_metrics"]["f1"]
        partial = r["match"]["partial_match"]
        grounded = r["grounded"]
        grounded_str = "" if grounded is True else ("" if grounded is False else "—")
        refusal_str = "—" if not r["should_refuse"] else ("" if r["refusal_correct"] else "")
        lines.append(
            f"| {r['id']} | {r['topic']} | {cite_f1:.2f} | {partial:.2f} | "
            f"{grounded_str} | {refusal_str} | {r['latency_ms']:.0f} |"
        )

    lines.extend([
        "",
        "## Failures",
        "",
    ])

    failures = [
        r for r in results["per_question"]
        if (r["citation_metrics"]["f1"] < 0.5 and not r["should_refuse"])
        or (r["should_refuse"] and not r["refusal_correct"])
        or (r["grounded"] is False)
    ]

    if not failures:
        lines.append("_No failures._")
    else:
        for r in failures:
            lines.append(f"### {r['id']} — {r['question']}")
            lines.append("")
            lines.append(f"**Topic:** {r['topic']} · **Should refuse:** {r['should_refuse']}")
            lines.append("")
            lines.append(f"**Answer:** {r['answer']}")
            lines.append("")
            if r["citation_metrics"]["false_positives"]:
                lines.append(f"- Invented citations: `{', '.join(r['citation_metrics']['false_positives'])}`")
            if r["citation_metrics"]["false_negatives"]:
                lines.append(f"- Missing expected citations: `{', '.join(r['citation_metrics']['false_negatives'])}`")
            if r["match"]["missed"]:
                lines.append(f"- Missing required substrings: `{r['match']['missed']}`")
            if r["grounded"] is False:
                lines.append(f"- Judge said not grounded: _{r['groundedness_rationale']}_")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval] Wrote markdown summary to {output_path}")


# ----- CLI -----

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the RAG evaluation harness.")
    p.add_argument("--eval-set", type=Path, default=_DEFAULT_EVAL_SET,
                   help="Path to eval_set.json")
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON path (default: eval/results/run-<timestamp>.json)")
    p.add_argument("--markdown-output", type=Path, default=None,
                   help="Output markdown summary path (default: alongside JSON)")
    p.add_argument("--no-judge", action="store_true",
                   help="Skip the LLM-as-judge groundedness check (faster)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        _DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        args.output = _DEFAULT_RESULTS_DIR / f"run-{ts}.json"

    if args.markdown_output is None:
        args.markdown_output = args.output.with_suffix(".md")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    results = run_evaluation(
        eval_set_path=args.eval_set,
        judge_enabled=not args.no_judge,
    )
    elapsed = time.perf_counter() - t0

    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[eval] Wrote results to {args.output}")

    write_markdown_summary(results, args.markdown_output)

    # Print a tight summary to stdout for CI logs
    agg = results["aggregates"]
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Groundedness:     {agg['groundedness_rate']:.1%}")
    print(f"Citation F1:      {agg['citation_f1_mean']:.3f}")
    print(f"Partial match:    {agg['partial_match_mean']:.3f}")
    print(f"Refusal rate:     {agg['refusal_rate']:.1%}  (on {agg['n_refusal']} refusal Qs)")
    print(f"Latency p50/p95:  {agg['latency']['p50']:.0f}ms / {agg['latency']['p95']:.0f}ms")
    print(f"Total elapsed:    {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
