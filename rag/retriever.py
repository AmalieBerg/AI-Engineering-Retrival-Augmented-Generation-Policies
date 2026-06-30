"""Retrieval layer for the RAG pipeline.

Provides a Retriever that:
  1. Loads the persisted Chroma collection on init
  2. Runs top-k search with optional MMR diversity
  3. Optionally re-ranks results using a cross-encoder

The Retriever is constructed once per process and is thread-safe for reads.
"""
from __future__ import annotations

from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings
from ingest.embedder import get_embedding_model
from rag.types import RetrievedChunk


def _chunk_from_document(doc: Document, score: float = 0.0) -> RetrievedChunk:
    """Convert a LangChain Document into a RetrievedChunk."""
    m = doc.metadata
    return RetrievedChunk(
        content=doc.page_content,
        doc_id=m.get("doc_id", "UNKNOWN"),
        doc_title=m.get("doc_title", ""),
        source_path=m.get("source_path", ""),
        section=m.get("section_path") or m.get("h2") or m.get("h1"),
        chunk_index=int(m.get("chunk_index", 0)),
        score=score,
    )


class Retriever:
    """Wraps a persisted Chroma collection with optional reranking."""

    def __init__(
        self,
        chroma_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        use_mmr: Optional[bool] = None,
        use_reranker: Optional[bool] = None,
    ):
        self.chroma_dir = chroma_dir or str(settings.chroma_dir)
        self.collection_name = collection_name or settings.chroma_collection
        self.embedding_model_name = embedding_model_name or settings.embedding_model
        self.use_mmr = settings.use_mmr if use_mmr is None else use_mmr
        self.use_reranker = settings.use_reranker if use_reranker is None else use_reranker

        embeddings = get_embedding_model(self.embedding_model_name)
        self._store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.chroma_dir,
            embedding_function=embeddings,
        )

        # Lazy-load the reranker only if enabled, since it pulls in a separate model
        self._reranker = None
        if self.use_reranker:
            self._reranker = self._load_reranker()

    @staticmethod
    def _load_reranker():
        """Load the cross-encoder reranker.

        Defaults to ms-marco-MiniLM-L-6-v2 (~80 MB, runs on CPU in tens of ms).
        Returns None if sentence-transformers is unavailable, falling back to
        first-pass scoring only.
        """
        try:
            from sentence_transformers import CrossEncoder
            return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"[Retriever] Reranker unavailable, falling back to first-pass only: {e}")
            return None

    def count(self) -> int:
        """Number of indexed chunks. Useful for health checks."""
        return self._store._collection.count()

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Run retrieval for `query` and return the top-k chunks.

        Pipeline:
          1. Vector search: fetch top fetch_k by cosine/L2 similarity
          2. (Optional) MMR diversification: re-pick k from fetch_k for diversity
          3. (Optional) Cross-encoder rerank: re-score with a more accurate model
          4. Truncate to k
        """
        k = k if k is not None else settings.retrieval_k
        fetch_k = fetch_k if fetch_k is not None else settings.fetch_k

        if not query or not query.strip():
            return []

        # If reranking is enabled, we want a larger candidate pool to rerank from.
        # Otherwise just use MMR or plain similarity.
        candidate_pool_size = max(fetch_k, k * 4) if self.use_reranker else fetch_k

        # Step 1+2: first-pass retrieval (with MMR if enabled)
        if self.use_mmr:
            # MMR doesn't return scores natively — we'll fill them in as 0 and
            # let the reranker overwrite if it runs.
            docs = self._store.max_marginal_relevance_search(
                query,
                k=candidate_pool_size if self.use_reranker else k,
                fetch_k=candidate_pool_size,
                lambda_mult=0.5,  # 0=max diversity, 1=max relevance
            )
            candidates = [_chunk_from_document(d, score=0.0) for d in docs]
        else:
            # Plain similarity search with scores (lower = closer in L2)
            results = self._store.similarity_search_with_score(query, k=candidate_pool_size)
            candidates = [
                _chunk_from_document(d, score=float(score))
                for d, score in results
            ]

        # Step 3: cross-encoder rerank (if enabled and loaded)
        if self.use_reranker and self._reranker is not None and candidates:
            pairs = [[query, c.content] for c in candidates]
            scores = self._reranker.predict(pairs)
            # Higher score = more relevant for cross-encoders
            scored = list(zip(candidates, scores))
            scored.sort(key=lambda x: float(x[1]), reverse=True)
            candidates = [
                RetrievedChunk(
                    content=c.content,
                    doc_id=c.doc_id,
                    doc_title=c.doc_title,
                    source_path=c.source_path,
                    section=c.section,
                    chunk_index=c.chunk_index,
                    score=float(s),
                )
                for c, s in scored
            ]

        return candidates[:k]
