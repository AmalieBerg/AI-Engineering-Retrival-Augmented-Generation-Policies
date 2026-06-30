"""Ablation runner: sweep configurations and compare results.

Re-runs the evaluation under multiple configurations to demonstrate that the
chosen defaults are well-considered. Each ablation gets its own results file,
and a summary table is written comparing all runs on the headline metrics.

Usage:
  python -m eval.run_ablations
  python -m eval.run_ablations --skip-rebuild   # if chunking unchanged
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.run_eval import run_evaluation, write_markdown_summary  # noqa: E402


_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "ablations"


# Each ablation is a name + env var overrides + (optional) rebuild flag.
# When chunk_size/overlap change, we must rebuild the index.
ABLATIONS: List[Dict] = [
    # === Baseline ===
    {"name": "baseline", "env": {}, "rebuild": False},

    # === Retrieval k sweep ===
    {"name": "k3", "env": {"RETRIEVAL_K": "3"}, "rebuild": False},
    {"name": "k8", "env": {"RETRIEVAL_K": "8"}, "rebuild": False},

    # === Chunk size sweep (requires rebuild) ===
    {"name": "chunk400", "env": {"CHUNK_SIZE": "400", "CHUNK_OVERLAP": "80"}, "rebuild": True},
    {"name": "chunk1200", "env": {"CHUNK_SIZE": "1200", "CHUNK_OVERLAP": "240"}, "rebuild": True},

    # === Reranker on ===
    {"name": "reranker", "env": {"USE_RERANKER": "true"}, "rebuild": False},

    # === MMR off (pure similarity search) ===
    {"name": "no_mmr", "env": {"USE_MMR": "false"}, "rebuild": False},
]


def rebuild_index() -> None:
    """Rebuild the Chroma index using the current env settings."""
    print("[ablation] Rebuilding index...")
    result = subprocess.run(
        [sys.executable, "-m", "ingest.build_index", "--rebuild"],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("Index rebuild failed")
    print("[ablation] Rebuild complete")


def run_one_ablation(ablation: Dict, eval_set_path: Path, results_dir: Path,
                     judge_enabled: bool, skip_rebuild: bool) -> Path:
    """Run one ablation and write its results JSON. Returns the JSON path."""
    name = ablation["name"]
    env_overrides = ablation["env"]
    needs_rebuild = ablation.get("rebuild", False)

    print(f"\n{'=' * 60}\nABLATION: {name}\n{'=' * 60}")
    print(f"Env overrides: {env_overrides or '(none, using defaults)'}")

    # Apply env overrides for this run
    original_env = {k: os.environ.get(k) for k in env_overrides}
    for k, v in env_overrides.items():
        os.environ[k] = v

    # Force reload of settings (config.py caches them at import time)
    import importlib
    import config as config_module
    importlib.reload(config_module)

    try:
        if needs_rebuild and not skip_rebuild:
            rebuild_index()

        # Re-import after env changes so the right settings flow through
        for mod_name in ["rag.retriever", "rag.pipeline", "eval.run_eval"]:
            if mod_name in sys.modules:
                importlib.reload(sys.modules[mod_name])
        from eval.run_eval import run_evaluation as run_eval_fresh, write_markdown_summary as md_fresh

        results = run_eval_fresh(eval_set_path=eval_set_path, judge_enabled=judge_enabled)
        results["meta"]["ablation_name"] = name
        results["meta"]["ablation_env_overrides"] = env_overrides

        json_path = results_dir / f"{name}.json"
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        md_fresh(results, results_dir / f"{name}.md")
        print(f"[ablation] {name}  {json_path}")
        return json_path
    finally:
        # Restore original env values for subsequent ablations
        for k, v in original_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def write_comparison_table(result_paths: List[Path], output_path: Path) -> None:
    """Write a side-by-side comparison of all ablations on the headline metrics."""
    rows = []
    for p in result_paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        agg = data["aggregates"]
        cfg = data["meta"]["config"]
        rows.append({
            "name": data["meta"].get("ablation_name", p.stem),
            "k": cfg["retrieval_k"],
            "chunk": cfg["chunk_size"],
            "mmr": cfg["use_mmr"],
            "rerank": cfg["use_reranker"],
            "groundedness": agg["groundedness_rate"],
            "cite_f1": agg["citation_f1_mean"],
            "partial": agg["partial_match_mean"],
            "refusal": agg["refusal_rate"],
            "p50": agg["latency"]["p50"],
            "p95": agg["latency"]["p95"],
        })

    lines = [
        "# Ablation Comparison",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat()}_",
        "",
        "Each row is a single evaluation run under one configuration variant. Bold "
        "values indicate the best score in each column.",
        "",
        "| Variant | k | chunk | MMR | rerank | Grounded | Cite F1 | Partial | Refusal | p50 ms | p95 ms |",
        "|---------|---|-------|-----|--------|----------|---------|---------|---------|--------|--------|",
    ]

    def best(col: str, higher_is_better: bool = True) -> float:
        vals = [r[col] for r in rows]
        return max(vals) if higher_is_better else min(vals)

    bests = {
        "groundedness": best("groundedness", True),
        "cite_f1": best("cite_f1", True),
        "partial": best("partial", True),
        "refusal": best("refusal", True),
        "p50": best("p50", False),
        "p95": best("p95", False),
    }

    def fmt(val: float, key: str, suffix: str = "") -> str:
        is_best = abs(val - bests[key]) < 1e-9
        s = f"{val:.1%}" if suffix == "%" else (f"{val:.3f}" if suffix == "" else f"{val:.0f}")
        return f"**{s}**" if is_best else s

    for r in rows:
        lines.append(
            f"| {r['name']} | {r['k']} | {r['chunk']} | {r['mmr']} | {r['rerank']} | "
            f"{fmt(r['groundedness'], 'groundedness', '%')} | "
            f"{fmt(r['cite_f1'], 'cite_f1')} | "
            f"{fmt(r['partial'], 'partial')} | "
            f"{fmt(r['refusal'], 'refusal', '%')} | "
            f"{fmt(r['p50'], 'p50', 'ms')} | "
            f"{fmt(r['p95'], 'p95', 'ms')} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[ablation] Wrote comparison to {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ablation sweep.")
    p.add_argument("--eval-set", type=Path,
                   default=Path(__file__).resolve().parent / "eval_set.json")
    p.add_argument("--no-judge", action="store_true",
                   help="Skip groundedness judging (faster)")
    p.add_argument("--skip-rebuild", action="store_true",
                   help="Skip chunk-size ablations that require rebuilding the index")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result_paths: List[Path] = []
    for ablation in ABLATIONS:
        if args.skip_rebuild and ablation.get("rebuild"):
            print(f"[ablation] Skipping {ablation['name']} (requires rebuild)")
            continue
        path = run_one_ablation(
            ablation=ablation,
            eval_set_path=args.eval_set,
            results_dir=_RESULTS_DIR,
            judge_enabled=not args.no_judge,
            skip_rebuild=args.skip_rebuild,
        )
        result_paths.append(path)

    write_comparison_table(result_paths, _RESULTS_DIR / "comparison.md")


if __name__ == "__main__":
    main()
