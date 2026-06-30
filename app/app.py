"""Flask web application for the policy RAG system.

Exposes three endpoints per the project spec:
  - GET  /         Chat UI (HTML)
  - POST /chat     Question -> answer + citations + latency (JSON)
  - GET  /health   Readiness/health check (JSON)

The pipeline is constructed once at app startup so the embedding model and
vector store stay in memory for the process lifetime.
"""
from __future__ import annotations

import logging
from typing import Optional

from flask import Flask, jsonify, render_template, request

from config import settings
from rag.pipeline import RAGPipeline


logger = logging.getLogger(__name__)


def create_app(pipeline: Optional[RAGPipeline] = None) -> Flask:
    """Flask application factory.

    Accepts an optional pre-built pipeline for testing. In production we let
    the factory build it lazily on first request (to keep import-time fast
    for the CI smoke test).
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["JSON_SORT_KEYS"] = False

    # Holds the lazily-initialized pipeline. We use a list so the closure can
    # mutate it without `nonlocal` gymnastics across multiple route handlers.
    _pipeline_holder: list[Optional[RAGPipeline]] = [pipeline]

    def get_pipeline() -> RAGPipeline:
        if _pipeline_holder[0] is None:
            logger.info("Initializing RAG pipeline on first request...")
            _pipeline_holder[0] = RAGPipeline()
            logger.info("RAG pipeline ready")
        return _pipeline_holder[0]  # type: ignore[return-value]

    # ----- Routes -----

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/chat")
    def chat():
        # Reject non-JSON bodies cleanly
        if not request.is_json:
            return jsonify({
                "error": "Request body must be JSON with Content-Type: application/json",
            }), 400

        data = request.get_json(silent=True) or {}
        question = data.get("question", "")

        if not isinstance(question, str):
            return jsonify({"error": "'question' must be a string"}), 400

        try:
            pipeline = get_pipeline()
            response = pipeline.answer(question)
        except Exception as e:
            # Hard failure (e.g., pipeline init crashed). Don't leak stack traces
            # to the client; log them for the operator.
            logger.exception("Unhandled error in /chat")
            return jsonify({
                "error": f"Internal error: {type(e).__name__}",
                "answer": "",
                "citations": [],
                "refused": True,
            }), 500

        return jsonify(response.to_user_dict())

    @app.get("/health")
    def health():
        """Lightweight readiness probe.

        Reports whether the pipeline can be constructed (vector index present)
        and whether the LLM API key is configured. Does NOT call the LLM --
        that would make health checks expensive and rate-limited.
        """
        status = {
            "status": "ok",
            "llm_key_configured": bool(settings.groq_api_key),
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
        }

        # Try to read the chunk count without forcing a full pipeline build
        try:
            pipeline = get_pipeline()
            chunk_count = pipeline.retriever.count()
            status["indexed_chunks"] = chunk_count
            if chunk_count == 0:
                status["status"] = "degraded"
                status["warning"] = "Vector index is empty; run `python -m ingest.build_index`"
        except Exception as e:
            status["status"] = "degraded"
            status["error"] = f"Pipeline init failed: {type(e).__name__}: {e}"

        http_code = 200 if status["status"] == "ok" else 503
        return jsonify(status), http_code

    # ----- Error handlers -----

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "Method not allowed"}), 405

    return app


# Convenience for `python -m app.app`
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = create_app()
    app.run(
        host=settings.flask_host,
        port=settings.flask_port,
        debug=settings.flask_debug,
    )


if __name__ == "__main__":
    main()
