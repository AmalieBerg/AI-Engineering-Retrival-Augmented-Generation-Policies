"""Flask app tests.

We use a stub pipeline injected via create_app() so these tests run without
needing Chroma, Groq, or any HuggingFace downloads.
"""
from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest

from app.app import create_app
from rag.pipeline import RAGResponse, Citation
from rag.prompt import REFUSAL_PHRASE


def make_response(
    answer: str = "PTO is 15 days [POL-HR-001].",
    citations: List[Citation] = None,
    refused: bool = False,
    error: str = None,
    latency_ms: float = 42.0,
) -> RAGResponse:
    return RAGResponse(
        answer=answer,
        citations=citations or [Citation(
            doc_id="POL-HR-001",
            doc_title="PTO Policy",
            source_path="md/pto.md",
            section="3. Accrual",
            snippet="Employees accrue 15 days per year.",
        )],
        latency_ms=latency_ms,
        refused=refused,
        error=error,
    )


def make_stub_pipeline(response: RAGResponse) -> MagicMock:
    pipeline = MagicMock()
    pipeline.answer.return_value = response
    # The /health endpoint reads retriever.count() — give it a non-zero value
    pipeline.retriever.count.return_value = 42
    return pipeline


@pytest.fixture
def client():
    pipeline = make_stub_pipeline(make_response())
    app = create_app(pipeline=pipeline)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# --- GET / -----------------------------------------------------------------

def test_index_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Northwind" in resp.data
    assert b"<form" in resp.data
    assert resp.content_type.startswith("text/html")


# --- POST /chat ------------------------------------------------------------

def test_chat_returns_answer_with_citations(client):
    resp = client.post("/chat", json={"question": "How much PTO?"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["answer"] == "PTO is 15 days [POL-HR-001]."
    assert len(data["citations"]) == 1
    assert data["citations"][0]["doc_id"] == "POL-HR-001"
    assert data["latency_ms"] == 42.0
    assert data["refused"] is False
    assert data["error"] is None


def test_chat_response_shape_omits_diagnostics(client):
    """The /chat response should not leak internal diagnostic fields."""
    resp = client.post("/chat", json={"question": "How much PTO?"})
    data = resp.get_json()
    assert set(data.keys()) == {"answer", "citations", "latency_ms", "refused", "error"}
    # Specifically, retrieved_chunks and guardrail_action stay internal
    assert "retrieved_chunks" not in data
    assert "guardrail_action" not in data


def test_chat_requires_json_content_type():
    pipeline = make_stub_pipeline(make_response())
    app = create_app(pipeline=pipeline)
    app.config["TESTING"] = True
    with app.test_client() as c:
        # Send form data instead of JSON
        resp = c.post("/chat", data="question=test")
        assert resp.status_code == 400
        assert "JSON" in resp.get_json()["error"]


def test_chat_rejects_non_string_question(client):
    resp = client.post("/chat", json={"question": 123})
    assert resp.status_code == 400


def test_chat_handles_missing_question_field():
    """An empty/missing question should produce a refusal via the pipeline guardrail."""
    pipeline = make_stub_pipeline(make_response(
        answer=REFUSAL_PHRASE,
        citations=[],
        refused=True,
    ))
    app = create_app(pipeline=pipeline)
    app.config["TESTING"] = True
    with app.test_client() as c:
        resp = c.post("/chat", json={})
        # The Flask layer accepts the request; the pipeline guardrail handles the empty string
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["refused"] is True


def test_chat_handles_pipeline_exception():
    """A hard pipeline failure should return 500 with a structured error, not a stack trace."""
    pipeline = MagicMock()
    pipeline.answer.side_effect = RuntimeError("simulated crash")
    pipeline.retriever.count.return_value = 42

    app = create_app(pipeline=pipeline)
    app.config["TESTING"] = True
    with app.test_client() as c:
        resp = c.post("/chat", json={"question": "anything"})
        assert resp.status_code == 500
        data = resp.get_json()
        assert "error" in data
        # Stack trace should not leak
        assert "simulated crash" not in data["error"]
        # The structured fields should still be present
        assert data["refused"] is True


# --- GET /health -----------------------------------------------------------

def test_health_returns_ok_when_indexed(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["indexed_chunks"] == 42
    assert "embedding_model" in data
    assert "llm_model" in data


def test_health_reports_degraded_with_empty_index():
    pipeline = make_stub_pipeline(make_response())
    pipeline.retriever.count.return_value = 0
    app = create_app(pipeline=pipeline)
    app.config["TESTING"] = True
    with app.test_client() as c:
        resp = c.get("/health")
        assert resp.status_code == 503
        data = resp.get_json()
        assert data["status"] == "degraded"
        assert "warning" in data


def test_health_reports_degraded_when_pipeline_init_fails():
    """If the pipeline can't be constructed, /health should still respond gracefully."""
    pipeline = MagicMock()
    # Access to .retriever raises (e.g., Chroma not yet built)
    type(pipeline).retriever = property(
        lambda self: (_ for _ in ()).throw(RuntimeError("no chroma_db dir"))
    )
    app = create_app(pipeline=pipeline)
    app.config["TESTING"] = True
    with app.test_client() as c:
        resp = c.get("/health")
        assert resp.status_code == 503
        data = resp.get_json()
        assert data["status"] == "degraded"
        assert "error" in data


# --- Error handlers --------------------------------------------------------

def test_404_returns_json(client):
    resp = client.get("/nonexistent-route")
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "Not found"


def test_405_returns_json(client):
    # /chat is POST-only; a GET should hit method-not-allowed
    resp = client.get("/chat")
    assert resp.status_code == 405
    assert resp.get_json()["error"] == "Method not allowed"
