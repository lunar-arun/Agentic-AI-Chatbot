"""
validation.py — input filtering and small UI-safety helpers.

Kept free of Streamlit imports so it can be unit-tested.
"""

import html
import re

MAX_MESSAGE_CHARS = 1000


def filter_input(text: str) -> str | None:
    """
    Return an error message if *text* should be rejected, or None if OK.
    Filtered messages never reach any agent, RAG, or history.
    """
    if not text or not text.strip():
        return "⚠️ Please enter a message."

    if len(text) > MAX_MESSAGE_CHARS:
        return f"⚠️ Message is too long (max {MAX_MESSAGE_CHARS:,} characters)."

    stripped = text.strip()

    # Repeated-character / emoji spam (single char repeated ≥8 times)
    if re.search(r"(.)\1{7,}", stripped, re.DOTALL):
        return "⚠️ That looks like spam. Please enter a real question."

    if len(set(stripped)) <= 2 and len(stripped) > 5:
        return "⚠️ That looks like spam. Please enter a real question."

    # URL with no other text
    if re.match(r"^https?://\S+$", stripped):
        return "⚠️ Please include a question along with the link."

    return None


def escape_html(text: str) -> str:
    """Escape text for safe use inside HTML rendered with unsafe_allow_html."""
    return html.escape(str(text), quote=True)


def inline_code_safe(text: str) -> str:
    """Make text safe to place inside a Markdown `inline code` span."""
    return str(text).replace("`", "'").replace("\n", " ")
