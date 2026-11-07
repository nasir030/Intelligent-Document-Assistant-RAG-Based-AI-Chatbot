"""Hugging Face sentence-transformer embedding factory."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=3)
def create_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Embeddings:
    """Create and reuse a local Sentence Transformers embedding model."""
    if not model_name.strip():
        raise ValueError("The embedding model name cannot be empty.")
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
