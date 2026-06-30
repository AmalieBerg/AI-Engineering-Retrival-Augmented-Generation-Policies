"""Central configuration for the RAG application.

Reads environment variables (with defaults) and exposes a single immutable
Settings object. All modules import from here rather than reading env vars
directly, so config changes happen in one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (if present)
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # LLM
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    llm_temperature: float = _env_float("LLM_TEMPERATURE", 0.1)
    max_answer_tokens: int = _env_int("MAX_ANSWER_TOKENS", 500)

    # Embedding
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # Paths
    project_root: Path = _PROJECT_ROOT
    corpus_dir: Path = Path(os.getenv("CORPUS_DIR", str(_PROJECT_ROOT / "corpus")))
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", str(_PROJECT_ROOT / "chroma_db")))
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "northwind_policies")

    # Chunking
    chunk_size: int = _env_int("CHUNK_SIZE", 800)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 150)

    # Retrieval
    retrieval_k: int = _env_int("RETRIEVAL_K", 5)
    fetch_k: int = _env_int("FETCH_K", 20)
    use_mmr: bool = _env_bool("USE_MMR", False)
    use_reranker: bool = _env_bool("USE_RERANKER", True)

    # Flask
    flask_host: str = os.getenv("FLASK_HOST", "0.0.0.0")
    flask_port: int = _env_int("FLASK_PORT", 8000)
    flask_debug: bool = _env_bool("FLASK_DEBUG", False)

    # Determinism
    random_seed: int = _env_int("RANDOM_SEED", 42)

    def require_groq_key(self) -> None:
        """Raise a clear error if the Groq API key is missing.

        Used by code paths that actually need to call the LLM. Lets the rest of
        the app (ingestion, retrieval) work without a key for development.
        """
        if not self.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Get a free key from https://console.groq.com and add it to .env"
            )


# Single shared settings instance
settings = Settings()
