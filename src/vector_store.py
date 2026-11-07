"""In-memory FAISS indexing and similarity retrieval."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStore:
    """A FAISS index built in memory for one uploaded-document collection."""

    def __init__(self) -> None:
        self._index: FAISS | None = None

    @property
    def is_empty(self) -> bool:
        return self._index is None

    @property
    def chunk_count(self) -> int:
        if self._index is None:
            return 0
        return self._index.index.ntotal

    def build(self, chunks: Sequence[Document], embeddings: Embeddings) -> None:
        """Replace the current index with a fresh index for the supplied chunks."""
        if not chunks:
            self._index = None
            return
        self._index = FAISS.from_documents(list(chunks), embeddings)

    def search(
        self,
        query: str,
        embeddings: Embeddings,
        top_k: int = 4,
        max_distance: float = 1.5,
    ) -> list[tuple[Document, float]]:
        """Return nearest chunks, filtering out results beyond the L2 distance."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if max_distance < 0:
            raise ValueError("max_distance must be non-negative.")
        if not query.strip() or self._index is None:
            return []

        matches = self._index.similarity_search_with_score(query, k=top_k)
        return [
            (document, float(distance))
            for document, distance in matches
            if float(distance) <= max_distance
        ]
