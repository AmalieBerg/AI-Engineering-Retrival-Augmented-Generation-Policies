"""Tests for ingest.loaders.

These tests run against the actual project corpus, since the loaders' contract
is tightly coupled to it (doc_id extraction depends on Northwind's header format).
"""
from __future__ import annotations

import pytest
from pathlib import Path

from config import settings
from ingest.loaders import load_corpus


@pytest.fixture(scope="module")
def documents():
    return load_corpus(settings.corpus_dir)


def test_corpus_loads_at_least_one_document_per_format(documents):
    formats = {d.metadata["file_format"] for d in documents}
    # Each format the corpus uses should have produced at least one Document
    assert "md" in formats
    assert "html" in formats
    assert "pdf" in formats
    assert "txt" in formats


def test_every_document_has_required_metadata(documents):
    required_keys = {"doc_id", "doc_title", "source_path", "file_format"}
    for d in documents:
        missing = required_keys - set(d.metadata)
        assert not missing, f"Document {d.metadata.get('source_path')} missing {missing}"


def test_doc_ids_follow_pol_format(documents):
    import re
    pattern = re.compile(r"^POL-[A-Z]{2,4}-\d{3}$")
    for d in documents:
        doc_id = d.metadata["doc_id"]
        assert pattern.match(doc_id), f"Bad doc_id format: {doc_id} from {d.metadata['source_path']}"


def test_all_15_unique_docs_are_present(documents):
    """The corpus has exactly 15 distinct policies; ingestion should find them all."""
    unique_doc_ids = {d.metadata["doc_id"] for d in documents}
    assert len(unique_doc_ids) == 15, (
        f"Expected 15 unique doc IDs, got {len(unique_doc_ids)}: {sorted(unique_doc_ids)}"
    )


def test_document_content_is_non_trivial(documents):
    """Every Document should have substantive content - guards against empty PDF pages."""
    for d in documents:
        # Allow short pages but flag completely empty content as a bug
        assert d.page_content.strip(), (
            f"Empty content in {d.metadata['source_path']}"
        )


def test_loader_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "does-not-exist")
