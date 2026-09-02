from __future__ import annotations

import re


SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"(cookie|sessionid|token|api[_-]?key)\s*[:=]\s*[^\s,;]+", re.IGNORECASE),
]


def sanitize_learning_text(text: str | None) -> str | None:
    if text is None:
        return None
    sanitized = text
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized


def normalize_required_text(value: str | None, field_name: str) -> str:
    normalized = " ".join((value or "").split())
    if not normalized:
        raise ValueError(f"Learning candidate requires {field_name}")
    return normalized
