"""Embedding model factory.

We default to BAAI/bge-small-en-v1.5 because it:
  - Is free (Apache 2.0 license)
  - Runs locally (no API costs, no rate limits)
  - Is small enough to fit in CI memory (~130 MB)
  - Performs strongly on the MTEB retrieval benchmark

The model is downloaded on first use and cached to ~/.cache/huggingface.
"""
from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5") -> HuggingFaceEmbeddings:
    """Return a LangChain embeddings wrapper for the given HuggingFace model.

    Encode kwargs:
      - normalize_embeddings=True: required for cosine similarity to behave well
        with Chroma's L2-distance default.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
