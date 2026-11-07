from src.chat_manager import export_chat_json, export_chat_text, new_message
from utils.helpers import MAX_UPLOAD_SIZE_BYTES, validate_upload


def test_upload_validation() -> None:
    assert validate_upload("document.pdf", 100) is None
    assert "Unsupported" in validate_upload("document.exe", 100)
    assert "empty" in validate_upload("document.txt", 0)
    assert "20 MB" in validate_upload("document.docx", MAX_UPLOAD_SIZE_BYTES + 1)


def test_chat_exports_include_content_and_sources() -> None:
    messages = [
        new_message("user", "What is in the file?"),
        new_message(
            "assistant",
            "A short answer.",
            sources=[
                {
                    "filename": "file.pdf",
                    "page": 2,
                    "passage": "Retrieved passage.",
                    "distance": 0.2,
                }
            ],
        ),
    ]

    assert "A short answer." in export_chat_text(messages)
    assert "file.pdf, page 2" in export_chat_text(messages)
    assert '"passage": "Retrieved passage."' in export_chat_json(messages)
