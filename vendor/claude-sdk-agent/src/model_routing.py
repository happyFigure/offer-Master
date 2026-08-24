from __future__ import annotations


_DEFAULT_MODEL_ALIASES = frozenset(
    {
        "",
        "claude-code",
        "openclaw:main",
    }
)


def resolve_effective_model(requested_model: object, default_model: str) -> str:
    requested = str(requested_model or "").strip()
    fallback = str(default_model or "").strip()
    if not requested:
        return fallback
    if requested.lower() in _DEFAULT_MODEL_ALIASES:
        return fallback
    return requested
