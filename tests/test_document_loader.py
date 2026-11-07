from io import BytesIO

import pytest
from docx import Document as DocxDocument
from reportlab.pdfgen import canvas

from src.document_loader import DocumentProcessingError, load_document


def make_pdf() -> bytes:
    output = BytesIO()
    pdf = canvas.Canvas(output)
    pdf.drawString(72, 720, "Page one content.")
    pdf.showPage()
    pdf.drawString(72, 720, "Page two content.")
    pdf.save()
    return output.getvalue()


def make_docx() -> bytes:
    output = BytesIO()
    document = DocxDocument()
    document.add_paragraph("DOCX paragraph content.")
    table = document.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "DOCX table content."
    document.save(output)
    return output.getvalue()


def test_extract_pdf_pages_with_page_metadata() -> None:
    documents = load_document("report.pdf", make_pdf())

    assert [document.metadata["page"] for document in documents] == [1, 2]
    assert all(document.metadata["source"] == "report.pdf" for document in documents)
    assert "Page one" in documents[0].page_content


def test_extract_txt_content() -> None:
    documents = load_document("notes.txt", b"\xef\xbb\xbfFirst line.\n\nSecond line.")

    assert len(documents) == 1
    assert documents[0].page_content == "First line.\n\nSecond line."
    assert documents[0].metadata == {"source": "notes.txt"}


def test_extract_docx_content() -> None:
    documents = load_document("notes.docx", make_docx())

    assert len(documents) == 1
    assert "DOCX paragraph content." in documents[0].page_content
    assert "DOCX table content." in documents[0].page_content
    assert documents[0].metadata["source"] == "notes.docx"


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("empty.txt", b" \n\t"),
        ("empty.pdf", b"not a pdf"),
        ("empty.docx", b"not a docx"),
        ("bad.txt", b"\xff"),
    ],
)
def test_empty_or_unreadable_documents_raise(filename: str, content: bytes) -> None:
    with pytest.raises(DocumentProcessingError):
        load_document(filename, content)


def test_unsupported_format_raises() -> None:
    with pytest.raises(DocumentProcessingError, match="Unsupported file type"):
        load_document("notes.csv", b"text")
