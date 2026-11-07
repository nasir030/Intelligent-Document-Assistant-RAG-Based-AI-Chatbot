import pytest
from langchain_core.documents import Document

from src.text_splitter import split_documents


def test_chunking_and_metadata_preservation() -> None:
    original = Document(
        page_content="alpha " * 400,
        metadata={"source": "paper.pdf", "page": 3},
    )

    chunks = split_documents([original], chunk_size=1000, chunk_overlap=200)

    assert len(chunks) > 1
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == [
        f"chunk-{index:06d}" for index in range(len(chunks))
    ]
    assert all(chunk.metadata["source"] == "paper.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 3 for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)


def test_empty_input_returns_no_chunks() -> None:
    assert split_documents([]) == []


def test_default_chunk_size_and_overlap() -> None:
    document = Document(page_content="a" * 2200)

    chunks = split_documents([document])

    assert [len(chunk.page_content) for chunk in chunks] == [1000, 1000, 600]
    assert chunks[0].page_content[-200:] == chunks[1].page_content[:200]
    assert chunks[1].page_content[-200:] == chunks[2].page_content[:200]


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (100, -1), (100, 100)],
)
def test_invalid_chunk_configuration(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_documents([Document(page_content="text")], size, overlap)
