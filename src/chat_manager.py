"""Helpers for serializing and downloading the Streamlit chat history."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def export_chat_json(messages: list[dict[str, Any]]) -> str:
    """Serialize conversation history as readable JSON."""
    return json.dumps(messages, ensure_ascii=False, indent=2)


def export_chat_text(messages: list[dict[str, Any]]) -> str:
    """Serialize conversation history and its source references as plain text."""
    lines: list[str] = []
    for message in messages:
        role = "You" if message.get("role") == "user" else "Assistant"
        timestamp = message.get("timestamp")
        lines.extend([f"{role} ({timestamp or 'unknown time'}):", message.get("content", "")])
        for source in message.get("sources", []):
            page = source.get("page")
            location = f", page {page}" if page is not None else ""
            lines.append(f"  Source: {source.get('filename', 'Unknown source')}{location}")
        lines.append("")
    return "\n".join(lines).strip()


def new_message(role: str, content: str, **extra: Any) -> dict[str, Any]:
    """Create a timestamped history message."""
    return {
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
