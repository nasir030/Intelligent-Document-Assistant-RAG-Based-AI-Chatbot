import sys
from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.embeddings import create_embeddings
from src.vector_store import VectorStore


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        return [1.0, 0.0] if "cats" in text.lower() else [0.0, 1.0]


def test_embedding_factory_uses_local_huggingface_model(monkeypatch) -> None:
    created = {}

    class FakeHuggingFaceEmbeddings:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langchain_huggingface",
        SimpleNamespace(HuggingFaceEmbeddings=FakeHuggingFaceEmbeddings),
    )
    from src.embeddings import create_embeddings

    create_embeddings.cache_clear()
    result = create_embeddings("test/local-model")
    create_embeddings.cache_clear()

    assert isinstance(result, FakeHuggingFaceEmbeddings)
    assert created["model_name"] == "test/local-model"
    assert created["encode_kwargs"]["normalize_embeddings"] is True


def test_faiss_index_creation_and_similarity_retrieval() -> None:
    embeddings = TinyEmbeddings()
    store = VectorStore()
    store.build(
        [
            Document(page_content="cats purr", metadata={"source": "cats.txt"}),
            Document(page_content="ocean tides", metadata={"source": "sea.txt"}),
        ],
        embeddings,
    )

    results = store.search("cats", embeddings, top_k=2, max_distance=0.5)

    assert store.chunk_count == 2
    assert len(results) == 1
    assert results[0][0].metadata["source"] == "cats.txt"


def test_empty_store_and_threshold_return_no_results() -> None:
    embeddings = TinyEmbeddings()
    store = VectorStore()

    assert store.search("cats", embeddings) == []
    store.build([Document(page_content="ocean tides")], embeddings)
    assert store.search("cats", embeddings, max_distance=0.1) == []


def test_building_empty_chunks_clears_index() -> None:
    store = VectorStore()
    store.build([Document(page_content="cats")], TinyEmbeddings())
    store.build([], TinyEmbeddings())

    assert store.is_empty
    assert store.chunk_count == 0
