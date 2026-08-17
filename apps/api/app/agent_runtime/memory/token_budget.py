from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any, Protocol


DEFAULT_RESERVE_TOKENS = 16384


class AgentMessageLike(Protocol):
    token_estimate: int | None
    content_text: str | None
    visible_content_text: str | None
    runtime_content_text: str | None
    content_json: dict[str, Any] | None


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0

    cjk_chars = 0
    ascii_non_space = 0
    other_non_space = 0
    for char in stripped:
        if char.isspace():
            continue
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF:
            cjk_chars += 1
        elif code < 128:
            ascii_non_space += 1
        else:
            other_non_space += 1

    token_estimate = cjk_chars + other_non_space + math.ceil(ascii_non_space / 4)
    return max(1, token_estimate)


def estimate_message_tokens(messages: Sequence[AgentMessageLike]) -> int:
    total = 0
    for message in messages:
        if message.token_estimate is not None and message.token_estimate >= 0:
            total += message.token_estimate
            continue

        total += estimate_tokens(_message_text(message))
        content_json = getattr(message, "content_json", None)
        if content_json:
            total += estimate_tokens(json.dumps(content_json, ensure_ascii=False, sort_keys=True))
    return total


def should_compact(
    *,
    context_tokens: int,
    context_window: int,
    reserve_tokens: int = DEFAULT_RESERVE_TOKENS,
) -> bool:
    return context_tokens > context_window - reserve_tokens


def _message_text(message: AgentMessageLike) -> str | None:
    return (
        getattr(message, "runtime_content_text", None)
        or getattr(message, "visible_content_text", None)
        or getattr(message, "content_text", None)
    )
