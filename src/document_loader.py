"""Load supported document formats into page-aware LangChain documents."""

from __future__ import annotations

import io
import re
from pathlib import Path

from docx import Document as DocxDocument
from langchain_core.documents import Document
from pypdf import PdfReader


class DocumentProcessingError(ValueError):
    """Raised when an uploaded document cannot be read or contains no text."""


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    return re.sub(r"[^\S\n]+", " ", text).strip()


def load_document(filename: str, content: bytes) -> list[Document]:
    """Extract text from one supported file, preserving page metadata when possible."""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".docx"}:
        raise DocumentProcessingError(
            f"Unsupported file type '{suffix or '(none)'}'. Upload PDF, TXT, or DOCX."
        )
    if not content:
        raise DocumentProcessingError("The uploaded file is empty.")

    documents: list[Document] = []
    try:
        if suffix == ".pdf":
            reader = PdfReader(io.BytesIO(content))
            for page_number, page in enumerate(reader.pages, start=1):
                text = _clean_text(page.extract_text() or "")
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": filename, "page": page_number},
                        )
                    )
        elif suffix == ".txt":
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise DocumentProcessingError(
                    "The TXT file is not valid UTF-8 text."
                ) from exc
            text = _clean_text(text)
            if text:
                documents.append(
                    Document(page_content=text, metadata={"source": filename})
                )
        else:
            docx = DocxDocument(io.BytesIO(content))
            paragraphs = [paragraph.text for paragraph in docx.paragraphs]
            tables = [
                "\n".join(
                    " | ".join(cell.text for cell in row.cells) for row in table.rows
                )
                for table in docx.tables
            ]
            text = _clean_text("\n".join([*paragraphs, *tables]))
            if text:
                documents.append(
                    Document(page_content=text, metadata={"source": filename})
                )
    except DocumentProcessingError:
        raise
    except Exception as exc:
        raise DocumentProcessingError(
            f"Could not read '{filename}': {exc}"
        ) from exc

    if not documents:
        raise DocumentProcessingError(
            f"No extractable text was found in '{filename}'."
        )
    return documents
