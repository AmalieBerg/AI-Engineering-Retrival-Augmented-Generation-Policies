"""Pure data types for the RAG pipeline.

Lives in its own module (no external imports) so prompt building, citation
parsing, and guardrail logic can be tested without bringing in heavyweight
dependencies like langchain or chromadb.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk with its similarity score and metadata.

    `score` semantics depend on the retrieval path:
      - L2 distance from Chroma similarity_search_with_score: lower is better
      - MMR (no native scores): 0.0 placeholder
      - Cross-encoder reranker: higher is better
    """

    content: str
    doc_id: str
    doc_title: str
    source_path: str
    section: Optional[str]
    chunk_index: int
    score: float
