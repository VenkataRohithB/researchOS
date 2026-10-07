"""Marking external content as untrusted data for the model."""

from __future__ import annotations

OPEN = "<<<UNTRUSTED_CONTENT"
CLOSE = "<<<END_UNTRUSTED_CONTENT>>>"


def untrusted(text: str, origin: str) -> str:
    """Wrap external text in delimiters the system prompt tells the model to treat as data.
    Delimiter look-alikes inside the text are neutralised so content cannot close the block."""
    cleaned = text.replace("<<<", "\u2039" * 3).replace(">>>", "\u203a" * 3)
    return f"{OPEN} origin={origin}>>>\n{cleaned}\n{CLOSE}"
