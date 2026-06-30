"""Build the Chroma vector index from the corpus.

Run via:  python -m ingest.build_index

This is intentionally a top-level command (not just a library function) so the
GitHub Actions workflow can verify ingestion works on every PR.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from langchain_chroma import Chroma

# Make the project root importable when run as `python -m ingest.build_index`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from ingest.chunker import chunk_documents  # noqa: E402
from ingest.embedder import get_embedding_model  # noqa: E402
from ingest.loaders import load_corpus  # noqa: E402


def build_index(
    corpus_dir: Path,
    chroma_dir: Path,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model_name: str,
    seed: int,
    rebuild: bool = False,
) -> None:
    """End-to-end index build."""
    # Determinism
    random.seed(seed)
    np.random.seed(seed)

    if rebuild and chroma_dir.exists():
        print(f"[1/5] Removing existing Chroma directory at {chroma_dir}")
        shutil.rmtree(chroma_dir)

    print(f"[1/5] Loading corpus from {corpus_dir}")
    t0 = time.perf_counter()
    documents = load_corpus(corpus_dir)
    print(f"      Loaded {len(documents)} document objects in {time.perf_counter() - t0:.2f}s")

    fmt_counts = Counter(d.metadata.get("file_format") for d in documents)
    for fmt, n in sorted(fmt_counts.items()):
        print(f"        {fmt}: {n}")

    doc_ids = sorted({d.metadata.get("doc_id", "?") for d in documents})
    print(f"      Unique document IDs: {len(doc_ids)} ({', '.join(doc_ids)})")

    print(f"[2/5] Chunking (size={chunk_size}, overlap={chunk_overlap})")
    t0 = time.perf_counter()
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"      Produced {len(chunks)} chunks in {time.perf_counter() - t0:.2f}s")

    avg_len = sum(len(c.page_content) for c in chunks) / max(len(chunks), 1)
    print(f"      Average chunk size: {avg_len:.0f} chars")

    print(f"[3/5] Loading embedding model: {embedding_model_name}")
    t0 = time.perf_counter()
    embeddings = get_embedding_model(embedding_model_name)
    print(f"      Embedding model ready in {time.perf_counter() - t0:.2f}s")

    print(f"[4/5] Embedding and storing in Chroma at {chroma_dir}")
    t0 = time.perf_counter()
    chroma_dir.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(chroma_dir),
    )
    print(f"      Stored in {time.perf_counter() - t0:.2f}s")

    print("[5/5] Verifying")
    count = vectorstore._collection.count()
    print(f"      Collection '{collection_name}' contains {count} embedded chunks")
    assert count == len(chunks), f"Mismatch: stored {count}, expected {len(chunks)}"
    print("\nIndex build complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Chroma vector index from the corpus.")
    parser.add_argument("--corpus-dir", type=Path, default=settings.corpus_dir)
    parser.add_argument("--chroma-dir", type=Path, default=settings.chroma_dir)
    parser.add_argument("--collection", type=str, default=settings.chroma_collection)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--embedding-model", type=str, default=settings.embedding_model)
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete existing index before rebuilding")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(
        corpus_dir=args.corpus_dir,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model_name=args.embedding_model,
        seed=args.seed,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
