"""Smoke tests for top-level imports.

These mirror the build/start check the CI workflow runs. They catch the
common 'broke an import somewhere' regression before the rest of the suite
even gets a chance to run.
"""
from __future__ import annotations


def test_config_imports():
    from config import settings
    assert settings is not None
    assert settings.embedding_model.startswith("BAAI/")


def test_ingest_modules_import():
    from ingest import loaders, chunker, embedder, build_index
    assert callable(loaders.load_corpus)
    assert callable(chunker.chunk_documents)
    assert callable(embedder.get_embedding_model)
    assert callable(build_index.build_index)
