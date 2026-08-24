from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


@dataclass(frozen=True, slots=True)
class RuntimeStreamEvent:
    kind: Literal["text", "tool", "task", "approval", "question", "hook", "meta", "command", "artifacts"]
    text: str = ""
    payload: Mapping[str, Any] | None = None
