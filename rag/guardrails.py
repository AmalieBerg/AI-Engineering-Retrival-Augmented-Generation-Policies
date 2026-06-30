"""Guardrails for the RAG pipeline.

Two layers:
  - Input guardrails: validate the user question before retrieval
  - Output guardrails: validate the LLM's answer before returning to the user

Output guardrails are particularly important for groundedness: if the model
returns an answer with no citations and no refusal, that's a strong hallucination
signal -- we replace it with a refusal to avoid misleading the user.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rag.prompt import REFUSAL_PHRASE, is_refusal, parse_citations


MAX_QUESTION_LENGTH = 1000  # Characters
MIN_QUESTION_LENGTH = 3
MAX_ANSWER_LENGTH = 4000  # Characters (~600-700 words)


@dataclass(frozen=True)
class InputValidation:
    """Result of input validation."""
    ok: bool
    reason: Optional[str] = None


def validate_question(question: str) -> InputValidation:
    """Check whether a question is safe and well-formed enough to process.

    We are intentionally permissive here -- the LLM and its system prompt do
    most of the policy-vs-non-policy filtering. We only block obviously
    malformed input.
    """
    if question is None:
        return InputValidation(False, "Missing 'question' field")

    if not isinstance(question, str):
        return InputValidation(False, "Question must be a string")

    stripped = question.strip()
    if len(stripped) < MIN_QUESTION_LENGTH:
        return InputValidation(False, "Question is too short")

    if len(stripped) > MAX_QUESTION_LENGTH:
        return InputValidation(
            False,
            f"Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters",
        )

    return InputValidation(True)


@dataclass(frozen=True)
class OutputCheck:
    """Result of post-generation validation.

    If ok=False, the pipeline should replace the answer with `replacement`
    (typically the canonical refusal phrase) before returning to the user.
    """
    ok: bool
    reason: Optional[str] = None
    replacement: Optional[str] = None


import re as _re

# Markers that suggest a factual claim worth citing (numbers, money, percentages)
_FACT_MARKER_RE = _re.compile(r"[\d$%]")


def _is_substantive(answer: str) -> bool:
    """Heuristic: would a careful reader expect a citation for this answer?

    True if EITHER the answer is over ~25 characters OR contains any digit /
    dollar sign / percent sign (a fact-bearing marker). This catches short but
    specific claims like "Up to 15 days." (14 chars but has a number).

    False for clearly-trivial responses like "Yes.", "No.", "Contact HR.".
    """
    stripped = answer.strip()
    if len(stripped) > 25:
        return True
    if _FACT_MARKER_RE.search(stripped):
        return True
    return False


def validate_answer(answer: str, had_retrieved_chunks: bool) -> OutputCheck:
    """Post-hoc validation of the LLM's answer.

    Rules:
      1. Empty answer -> replace with refusal
      2. Refusal answer -> always passes
      3. No chunks were retrieved but the answer isn't a refusal -> replace with refusal
      4. Substantive non-refusal answer with citations -> passes
      5. Substantive non-refusal answer without citations -> replace with refusal
      6. Excessively long answer -> truncate to MAX_ANSWER_LENGTH
    """
    if not answer or not answer.strip():
        return OutputCheck(
            ok=False,
            reason="Empty answer from model",
            replacement=REFUSAL_PHRASE,
        )

    if is_refusal(answer):
        return OutputCheck(ok=True)

    # If we had no context to ground the answer in, any non-refusal response
    # is a hallucination signal.
    if not had_retrieved_chunks:
        return OutputCheck(
            ok=False,
            reason="Non-refusal answer produced with no retrieved context",
            replacement=REFUSAL_PHRASE,
        )

    citations = parse_citations(answer)

    if _is_substantive(answer) and not citations:
        return OutputCheck(
            ok=False,
            reason="Substantive answer with no citations -- possible hallucination",
            replacement=REFUSAL_PHRASE,
        )

    if len(answer) > MAX_ANSWER_LENGTH:
        return OutputCheck(
            ok=False,
            reason=f"Answer exceeds {MAX_ANSWER_LENGTH} chars; truncated",
            replacement=answer[:MAX_ANSWER_LENGTH].rstrip() + "...",
        )

    return OutputCheck(ok=True)


def filter_invalid_citations(
    cited_doc_ids: list[str],
    valid_doc_ids: set[str],
) -> tuple[list[str], list[str]]:
    """Split cited doc IDs into (valid, invalid) based on whether they were retrieved.

    The model is instructed not to invent doc IDs, but we double-check here.
    Invalid citations are reported separately so they can be logged but don't
    pollute the user-facing citation list.
    """
    valid = [d for d in cited_doc_ids if d in valid_doc_ids]
    invalid = [d for d in cited_doc_ids if d not in valid_doc_ids]
    return valid, invalid
