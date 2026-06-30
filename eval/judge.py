"""LLM-as-judge for groundedness scoring.

We call a separate LLM (typically the same Groq model used for generation, or
a stronger one for higher-fidelity judging) with judge_prompt.txt to evaluate
whether the answer is fully supported by the retrieved context.

The judge prompt is engineered to return strict JSON. We parse it defensively
since LLMs occasionally wrap output in code fences or add stray text.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from rag.llm import LLMClient


_JUDGE_PROMPT_PATH = Path(__file__).resolve().parent / "judge_prompt.txt"


@dataclass(frozen=True)
class JudgeVerdict:
    """One groundedness verdict from the judge."""
    grounded: Optional[bool]  # None = parse failure
    rationale: str
    raw: str  # The full raw LLM response, for audit


def load_judge_prompt() -> str:
    """Load the judge prompt template from disk."""
    return _JUDGE_PROMPT_PATH.read_text(encoding="utf-8")


def format_context_for_judge(retrieved_chunks: List[dict]) -> str:
    """Render retrieved chunks as a flat block for the judge prompt.

    Mirrors the format the answering model saw, so the judge evaluates against
    the same evidence the model had access to.
    """
    if not retrieved_chunks:
        return "(no chunks retrieved)"
    blocks = []
    for c in retrieved_chunks:
        label = f"[{c['doc_id']}"
        if c.get("doc_title"):
            label += f" - {c['doc_title']}"
        if c.get("section"):
            label += f" § {c['section']}"
        label += "]"
        blocks.append(f"{label}\n{c['content'].strip()}")
    return "\n\n".join(blocks)


# Pattern to extract a JSON object from the judge's output, even if wrapped in
# ```json fences or surrounded by stray prose.
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\"grounded\"[^{}]*\}", re.DOTALL)


def parse_verdict(raw_output: str) -> JudgeVerdict:
    """Parse the judge's JSON verdict defensively.

    Tolerates code fences (```json ... ```), trailing prose, single quotes, and
    Python booleans (True/False). Returns grounded=None if no verdict can be
    extracted at all.
    """
    if not raw_output:
        return JudgeVerdict(grounded=None, rationale="Empty judge output", raw=raw_output)

    text = raw_output.strip()

    # Strip code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    # First attempt: parse the whole thing as JSON
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Second attempt: find a JSON object containing "grounded" anywhere in the text
        match = _JSON_OBJECT_RE.search(text)
        if not match:
            return JudgeVerdict(
                grounded=None,
                rationale=f"Could not parse JSON from judge: {text[:200]}",
                raw=raw_output,
            )
        candidate = match.group(0)
        # Coerce Python literals if the model produced them
        candidate = candidate.replace("True", "true").replace("False", "false")
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as e:
            return JudgeVerdict(
                grounded=None,
                rationale=f"JSON parse failed: {e}",
                raw=raw_output,
            )

    grounded = obj.get("grounded")
    if not isinstance(grounded, bool):
        # Tolerate string variants
        if isinstance(grounded, str):
            grounded = grounded.strip().lower() in {"true", "yes", "1"}
        else:
            return JudgeVerdict(
                grounded=None,
                rationale=f"'grounded' field not boolean: {grounded!r}",
                raw=raw_output,
            )

    rationale = obj.get("rationale", "")
    return JudgeVerdict(grounded=grounded, rationale=str(rationale), raw=raw_output)


class GroundednessJudge:
    """Wraps an LLM client to produce groundedness verdicts."""

    def __init__(self, llm: LLMClient, prompt_template: Optional[str] = None):
        self.llm = llm
        self.prompt_template = prompt_template or load_judge_prompt()

    def judge(self, question: str, retrieved_chunks: List[dict], answer: str) -> JudgeVerdict:
        """Run one groundedness judgment.

        Returns a JudgeVerdict whose `grounded` field is True/False/None.
        Callers should treat grounded=None as "judge failed" and either skip
        or mark the question as unscored.
        """
        context = format_context_for_judge(retrieved_chunks)
        prompt = (self.prompt_template
                  .replace("{question}", question)
                  .replace("{context}", context)
                  .replace("{answer}", answer))

        try:
            # Judge calls use temperature=0 for determinism
            raw = self.llm.generate(
                system_prompt="You are a strict, fair evaluation judge.",
                user_prompt=prompt,
                max_tokens=300,
                temperature=0.0,
            )
        except Exception as e:
            return JudgeVerdict(grounded=None, rationale=f"Judge LLM call failed: {e}", raw="")

        return parse_verdict(raw)
