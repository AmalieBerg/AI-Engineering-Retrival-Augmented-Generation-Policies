"""Integration tests for the RAG pipeline with a mock LLM.

We mock both the LLM and the retriever so these tests are deterministic and
need neither a Groq key nor a built Chroma index. The point is to verify the
orchestration logic: prompt assembly, citation extraction, guardrail enforcement,
and response shape.
"""
from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock

from rag.pipeline import RAGPipeline, RAGResponse
from rag.prompt import REFUSAL_PHRASE
from rag.types import RetrievedChunk


class StubLLM:
    """A canned-response LLM for tests."""

    def __init__(self, response: str):
        self.response = response
        self.calls: List[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


def make_chunk(doc_id: str, content: str, doc_title: str = "Test Doc",
               section: str = "1. Intro") -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        doc_id=doc_id,
        doc_title=doc_title,
        source_path=f"md/{doc_id.lower()}.md",
        section=section,
        chunk_index=0,
        score=0.1,
    )


def make_pipeline(llm_response: str, retrieved_chunks: List[RetrievedChunk]) -> RAGPipeline:
    """Build a pipeline with a stub retriever and stub LLM."""
    retriever = MagicMock()
    retriever.retrieve.return_value = retrieved_chunks
    return RAGPipeline(retriever=retriever, llm=StubLLM(llm_response))


# --- Happy path ------------------------------------------------------------

def test_pipeline_returns_grounded_answer_with_citations():
    chunks = [make_chunk("POL-HR-001",
                         "Employees accrue 15 days of PTO per year for the first 2 years.")]
    pipe = make_pipeline(
        llm_response="Employees accrue 15 days of PTO per year [POL-HR-001].",
        retrieved_chunks=chunks,
    )
    resp = pipe.answer("How much PTO do I get?")

    assert isinstance(resp, RAGResponse)
    assert "15 days" in resp.answer
    assert len(resp.citations) == 1
    assert resp.citations[0].doc_id == "POL-HR-001"
    assert "15 days of PTO" in resp.citations[0].snippet
    assert not resp.refused
    assert resp.error is None
    assert resp.guardrail_action == "ok"
    assert resp.latency_ms > 0


def test_pipeline_extracts_multiple_citations():
    chunks = [
        make_chunk("POL-IT-003", "Source code may not be accessed from personal devices."),
        make_chunk("POL-SEC-001", "Personal devices may only access company data through approved methods."),
    ]
    pipe = make_pipeline(
        llm_response="Source code cannot be accessed from personal devices [POL-IT-003][POL-SEC-001].",
        retrieved_chunks=chunks,
    )
    resp = pipe.answer("Can I view source code on my phone?")
    assert len(resp.citations) == 2
    assert {c.doc_id for c in resp.citations} == {"POL-IT-003", "POL-SEC-001"}


# --- Refusal handling ------------------------------------------------------

def test_pipeline_passes_through_canonical_refusal():
    """When the LLM correctly refuses, the pipeline returns refused=True."""
    pipe = make_pipeline(llm_response=REFUSAL_PHRASE, retrieved_chunks=[])
    resp = pipe.answer("What is the company dress code?")
    assert resp.refused is True
    assert resp.answer == REFUSAL_PHRASE
    assert resp.citations == []
    assert resp.guardrail_action == "ok"


def test_pipeline_handles_empty_question():
    pipe = make_pipeline(llm_response="ignored", retrieved_chunks=[])
    resp = pipe.answer("")
    assert resp.refused
    assert resp.guardrail_action == "input_validation_failed"


def test_pipeline_handles_whitespace_only_question():
    pipe = make_pipeline(llm_response="ignored", retrieved_chunks=[])
    resp = pipe.answer("   \n  ")
    assert resp.refused
    assert resp.guardrail_action == "input_validation_failed"


# --- Output guardrails -----------------------------------------------------

def test_pipeline_replaces_substantive_answer_with_no_citations():
    """A hallucinated-looking answer gets replaced with a refusal."""
    chunks = [make_chunk("POL-HR-001", "PTO info here.")]
    pipe = make_pipeline(
        llm_response="The CEO of the company is John Smith and was hired in 2020.",
        retrieved_chunks=chunks,
    )
    resp = pipe.answer("Who is the CEO?")
    assert resp.answer == REFUSAL_PHRASE
    assert resp.refused
    assert resp.guardrail_action and "output_replaced" in resp.guardrail_action


def test_pipeline_filters_invalid_citations():
    """If the LLM cites a doc that wasn't retrieved, it's tracked but not surfaced."""
    chunks = [make_chunk("POL-HR-001", "Real chunk content.")]
    pipe = make_pipeline(
        llm_response="The answer is 15 [POL-HR-001][POL-FAKE-999].",
        retrieved_chunks=chunks,
    )
    resp = pipe.answer("How much PTO?")
    # Only the valid citation surfaces to the user
    assert [c.doc_id for c in resp.citations] == ["POL-HR-001"]
    # But the invalid one is tracked in diagnostics
    assert resp.invalid_citations == ["POL-FAKE-999"]


# --- Error handling --------------------------------------------------------

def test_pipeline_handles_llm_failure():
    """An LLM exception becomes a refusal, not a stack trace."""
    chunks = [make_chunk("POL-HR-001", "Content")]
    retriever = MagicMock()
    retriever.retrieve.return_value = chunks

    failing_llm = MagicMock()
    failing_llm.generate.side_effect = RuntimeError("simulated API outage")

    pipe = RAGPipeline(retriever=retriever, llm=failing_llm)
    resp = pipe.answer("How much PTO?")
    assert resp.refused
    assert resp.error is not None
    assert "simulated API outage" in resp.error
    assert resp.guardrail_action == "llm_error"


def test_pipeline_handles_retriever_failure():
    retriever = MagicMock()
    retriever.retrieve.side_effect = RuntimeError("chroma is down")
    pipe = RAGPipeline(retriever=retriever, llm=StubLLM("ignored"))
    resp = pipe.answer("How much PTO?")
    assert resp.refused
    assert resp.error is not None
    assert "chroma is down" in resp.error
    assert resp.guardrail_action == "retrieval_error"


# --- Response shape --------------------------------------------------------

def test_to_user_dict_omits_diagnostics():
    chunks = [make_chunk("POL-HR-001", "PTO is 15 days.")]
    pipe = make_pipeline("15 days [POL-HR-001].", chunks)
    resp = pipe.answer("How much PTO?")
    user_dict = resp.to_user_dict()
    assert set(user_dict.keys()) == {"answer", "citations", "latency_ms", "refused", "error"}
    # Diagnostics should not leak
    assert "retrieved_chunks" not in user_dict
    assert "invalid_citations" not in user_dict
    assert "guardrail_action" not in user_dict


def test_to_eval_dict_includes_diagnostics():
    chunks = [make_chunk("POL-HR-001", "PTO is 15 days.")]
    pipe = make_pipeline("15 days [POL-HR-001].", chunks)
    resp = pipe.answer("How much PTO?", include_chunks_in_response=True)
    eval_dict = resp.to_eval_dict()
    assert "retrieved_chunks" in eval_dict
    assert len(eval_dict["retrieved_chunks"]) == 1
    assert "invalid_citations" in eval_dict


def test_citations_have_truncated_snippets():
    long_content = "x" * 1000
    chunks = [make_chunk("POL-HR-001", long_content)]
    pipe = make_pipeline("Answer [POL-HR-001].", chunks)
    resp = pipe.answer("Question?")
    assert len(resp.citations) == 1
    # Snippet is capped (with ellipsis) so the UI can render it cleanly
    assert len(resp.citations[0].snippet) <= 300
    assert resp.citations[0].snippet.endswith("...")
