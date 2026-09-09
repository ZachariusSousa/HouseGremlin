from __future__ import annotations

import re


_URL = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_BEARER = re.compile(r"\bBearer\s+[^\s,;]+", re.IGNORECASE)
_NAMED_SECRET = re.compile(
    r"\b(api[_-]?key|access[_-]?token|token|secret|password|authorization)\b"
    r"\s*[:=]\s*[^\s,;&]+",
    re.IGNORECASE,
)
_OPENAI_STYLE_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")


def sanitize_public_text(
    value: object,
    *,
    max_length: int = 500,
    fallback: str = "unavailable",
) -> str:
    """Remove common credential carriers and return bounded browser-safe text."""
    text = " ".join(str(value).split())
    text = _URL.sub("[redacted-url]", text)
    text = _BEARER.sub("Bearer [redacted]", text)
    text = _NAMED_SECRET.sub(
        lambda match: f"{match.group(1)}=[redacted]",
        text,
    )
    text = _OPENAI_STYLE_KEY.sub("[redacted-token]", text)
    text = text.strip() or fallback
    if len(text) > max_length:
        text = text[: max_length - 1].rstrip() + "…"
    return text
