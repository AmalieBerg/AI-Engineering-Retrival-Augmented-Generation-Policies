"""RAGPipeline: end-to-end orchestration of retrieval and generation.

Composes retriever, prompt builder, LLM, and guardrails into a single
`answer()` call. This is the entrypoint used by both the Flask app and the
evaluation harness, so any improvement here lifts both.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from config import settings
from rag.guardrails import (
    filter_invalid_citations,
    validate_answer,
    validate_question,
)
from rag.llm import GroqLLMClient, LLMClient
from rag.prompt import (
    REFUSAL_PHRASE,
    build_system_prompt,
    build_user_prompt,
    is_refusal,
    parse_citations,
)
from rag.retriever import Retriever
from rag.types import RetrievedChunk


@dataclass(frozen=True)
class Citation:
    """A citation surfaced to the user for one supporting source."""
    doc_id: str
    doc_title: str
    source_path: str
    section: Optional[str]
    snippet: str


@dataclass
class RAGResponse:
    """Full structured response from the pipeline."""
    answer: str
    citations: List[Citation]
    latency_ms: float
    refused: bool
    error: Optional[str] = None
    # Diagnostic fields (omitted from JSON by default; eval harness reads them)
    retrieved_chunks: List[dict] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    guardrail_action: Optional[str] = None

    def to_user_dict(self) -> dict:
        """JSON-safe representation suitable for the /chat endpoint."""
        return {
            "answer": self.answer,
            "citations": [asdict(c) for c in self.citations],
            "latency_ms": round(self.latency_ms, 2),
            "refused": self.refused,
            "error": self.error,
        }

    def to_eval_dict(self) -> dict:
        """Verbose representation used by the evaluation harness."""
        return {
            **self.to_user_dict(),
            "retrieved_chunks": self.retrieved_chunks,
            "invalid_citations": self.invalid_citations,
            "guardrail_action": self.guardrail_action,
        }


# Maximum number of characters to include in a citation snippet
SNIPPET_MAX_CHARS = 280


class RAGPipeline:
    """The end-to-end RAG pipeline."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm: Optional[LLMClient] = None,
    ):
        self.retriever = retriever or Retriever()
        self.llm = llm or GroqLLMClient()

    def _make_citations(
        self,
        cited_doc_ids: List[str],
        chunks: List[RetrievedChunk],
    ) -> List[Citation]:
        """For each cited doc_id, pick the best supporting chunk and build a Citation.

        "Best" means the top-ranked retrieved chunk for that doc (chunks come in
        rank order from the retriever).
        """
        # Map doc_id -> top-ranked chunk that came back for it
        chunks_by_doc: dict[str, RetrievedChunk] = {}
        for c in chunks:
            if c.doc_id not in chunks_by_doc:
                chunks_by_doc[c.doc_id] = c

        citations: List[Citation] = []
        for doc_id in cited_doc_ids:
            chunk = chunks_by_doc.get(doc_id)
            if chunk is None:
                # Shouldn't happen since invalid citations are filtered upstream,
                # but guard anyway.
                continue
            snippet = chunk.content.strip()
            if len(snippet) > SNIPPET_MAX_CHARS:
                snippet = snippet[:SNIPPET_MAX_CHARS].rstrip() + "..."
            citations.append(Citation(
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                source_path=chunk.source_path,
                section=chunk.section,
                snippet=snippet,
            ))
        return citations

    def answer(self, question: str, *, include_chunks_in_response: bool = False) -> RAGResponse:
        """Answer one question end-to-end.

        Args:
            question: The user's question.
            include_chunks_in_response: When True, the response includes the full
                retrieved-chunk objects in its diagnostic fields. Set by the
                evaluation harness; the public /chat endpoint leaves it False.
        """
        t0 = time.perf_counter()

        # === 1) Input guardrail ===
        check = validate_question(question)
        if not check.ok:
            return RAGResponse(
                answer=REFUSAL_PHRASE,
                citations=[],
                latency_ms=(time.perf_counter() - t0) * 1000,
                refused=True,
                error=check.reason,
                guardrail_action="input_validation_failed",
            )

        # === 2) Retrieve ===
        try:
            chunks = self.retriever.retrieve(question)
        except Exception as e:
            return RAGResponse(
                answer=REFUSAL_PHRASE,
                citations=[],
                latency_ms=(time.perf_counter() - t0) * 1000,
                refused=True,
                error=f"Retrieval failed: {e}",
                guardrail_action="retrieval_error",
            )

        # === 3) Build prompt and call LLM ===
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(question, chunks)

        try:
            raw_answer = self.llm.generate(system_prompt, user_prompt)
        except Exception as e:
            return RAGResponse(
                answer=REFUSAL_PHRASE,
                citations=[],
                latency_ms=(time.perf_counter() - t0) * 1000,
                refused=True,
                error=f"LLM generation failed: {e}",
                guardrail_action="llm_error",
                retrieved_chunks=[asdict_chunk(c) for c in chunks] if include_chunks_in_response else [],
            )

        # === 4) Output guardrail ===
        out_check = validate_answer(raw_answer, had_retrieved_chunks=bool(chunks))
        if not out_check.ok:
            final_answer = out_check.replacement or REFUSAL_PHRASE
            return RAGResponse(
                answer=final_answer,
                citations=[],
                latency_ms=(time.perf_counter() - t0) * 1000,
                refused=is_refusal(final_answer),
                error=None,
                guardrail_action=f"output_replaced: {out_check.reason}",
                retrieved_chunks=[asdict_chunk(c) for c in chunks] if include_chunks_in_response else [],
            )

        # === 5) Extract & validate citations ===
        cited_ids = parse_citations(raw_answer)
        valid_doc_ids = {c.doc_id for c in chunks}
        valid_cites, invalid_cites = filter_invalid_citations(cited_ids, valid_doc_ids)

        citations = self._make_citations(valid_cites, chunks)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return RAGResponse(
            answer=raw_answer,
            citations=citations,
            latency_ms=elapsed_ms,
            refused=is_refusal(raw_answer),
            error=None,
            guardrail_action="ok",
            retrieved_chunks=[asdict_chunk(c) for c in chunks] if include_chunks_in_response else [],
            invalid_citations=invalid_cites,
        )


def asdict_chunk(chunk: RetrievedChunk) -> dict:
    """Serialize a RetrievedChunk for the eval harness."""
    return {
        "content": chunk.content,
        "doc_id": chunk.doc_id,
        "doc_title": chunk.doc_title,
        "source_path": chunk.source_path,
        "section": chunk.section,
        "chunk_index": chunk.chunk_index,
        "score": chunk.score,
    }
