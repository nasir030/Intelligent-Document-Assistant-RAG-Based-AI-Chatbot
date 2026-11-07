"""Shared upload validation helpers."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_UPLOAD_SIZE_MB = 20
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


def validate_upload(filename: str, size: int) -> str | None:
    """Return a user-facing validation error, or None when the upload is valid."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return "Unsupported file type. Upload a PDF, TXT, or DOCX document."
    if size <= 0:
        return "The uploaded file is empty."
    if size > MAX_UPLOAD_SIZE_BYTES:
        return f"File exceeds the {MAX_UPLOAD_SIZE_MB} MB upload limit."
    return None
