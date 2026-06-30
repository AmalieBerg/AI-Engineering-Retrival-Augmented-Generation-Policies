"""Tests for ingest.chunker."""
from __future__ import annotations

import pytest

from config import settings
from ingest.chunker import chunk_documents
from ingest.loaders import load_corpus


@pytest.fixture(scope="module")
def chunks():
    docs = load_corpus(settings.corpus_dir)
    return chunk_documents(docs, chunk_size=settings.chunk_size,
                           chunk_overlap=settings.chunk_overlap)


def test_chunking_produces_chunks(chunks):
    # Sanity floor: 15 docs should yield far more than 15 chunks at chunk_size=800
    assert len(chunks) >= 30, f"Suspiciously few chunks: {len(chunks)}"


def test_chunks_respect_size_budget(chunks):
    """Chunks should be within a reasonable factor of the configured size.

    The markdown-header splitter may produce chunks slightly larger than the
    configured chunk_size when an entire section fits (and that's desirable -
    we don't want to split mid-section). But anything > 2x the configured size
    suggests a bug where over-large sections weren't sub-split.
    """
    max_allowed = settings.chunk_size * 2
    too_big = [c for c in chunks if len(c.page_content) > max_allowed]
    assert not too_big, (
        f"{len(too_big)} chunks exceed 2x chunk_size ({max_allowed}); "
        f"first offender has {len(too_big[0].page_content)} chars"
    )


def test_chunks_preserve_doc_metadata(chunks):
    """Every chunk should retain its parent document's identity."""
    for c in chunks:
        assert "doc_id" in c.metadata
        assert "doc_title" in c.metadata
        assert "source_path" in c.metadata
        assert "chunk_index" in c.metadata
        assert isinstance(c.metadata["chunk_index"], int)


def test_chunk_indices_are_per_document(chunks):
    """Within a single source_path, chunk_index should start at 0."""
    by_source: dict[str, list[int]] = {}
    for c in chunks:
        by_source.setdefault(c.metadata["source_path"], []).append(c.metadata["chunk_index"])

    for source, indices in by_source.items():
        assert min(indices) == 0, f"chunk_index doesn't start at 0 for {source}: {indices}"


def test_chunking_is_deterministic():
    """Same input + same config -> same chunks."""
    docs = load_corpus(settings.corpus_dir)
    a = chunk_documents(docs, chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap)
    b = chunk_documents(docs, chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap)
    assert len(a) == len(b)
    for ca, cb in zip(a, b):
        assert ca.page_content == cb.page_content


def test_all_doc_ids_appear_in_chunks(chunks):
    """Chunking should not silently drop any documents."""
    chunk_doc_ids = {c.metadata["doc_id"] for c in chunks}
    assert len(chunk_doc_ids) == 15, (
        f"Expected 15 unique doc IDs in chunks, got {len(chunk_doc_ids)}"
    )
