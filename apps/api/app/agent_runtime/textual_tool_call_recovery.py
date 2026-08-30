from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.agent_runtime.tool_registry import AgentToolRegistry


@dataclass(frozen=True)
class TextualToolCall:
    tool_name: str
    tool_input: dict[str, Any]
    raw_tool_name: str
    raw_content: str


_TEXTUAL_TOOL_CALL_RE = re.compile(
    r"^\s*(?:(?:\*\*)?OfferMaster\s+AI(?:\*\*)?\s*)?"
    r"(?:Tool\s*call|工具调用)\s*[:：]\s*"
    r"(?P<tool>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s*"
    r"(?P<arguments>\{.*\})?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_MULTILINE_TEXTUAL_TOOL_CALL_RE = re.compile(
    r"^\s*(?:(?:\*\*)?OfferMaster\s+AI(?:\*\*)?\s*)?"
    r"(?:Tool\s*call|工具调用)\s*[:：]\s*"
    r"(?P<tool>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s*\n+"
    r"(?:Arguments|参数)\s*[:：]\s*"
    r"(?P<arguments>\{.*\})\s*$",
    re.IGNORECASE | re.DOTALL,
)


def recover_textual_tool_call(content: str, *, registry: AgentToolRegistry) -> TextualToolCall | None:
    normalized = str(content or "").replace("\\_", "_").strip()
    if not normalized:
        return None
    alias_to_tool_name = _alias_to_tool_name(registry)
    for candidate in _textual_tool_call_candidates(normalized):
        match = _TEXTUAL_TOOL_CALL_RE.match(candidate) or _MULTILINE_TEXTUAL_TOOL_CALL_RE.match(candidate)
        if match is None:
            continue
        raw_tool_name = match.group("tool")
        tool_name = alias_to_tool_name.get(raw_tool_name, raw_tool_name)
        if registry.get(tool_name) is None:
            continue
        tool_input = _parse_tool_call_arguments(match.group("arguments"))
        if tool_input is None:
            continue
        return TextualToolCall(
            tool_name=tool_name,
            tool_input=tool_input,
            raw_tool_name=raw_tool_name,
            raw_content=candidate,
        )
    return None


def _alias_to_tool_name(registry: AgentToolRegistry) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for definition in registry.list_definitions():
        tool_name = definition.name
        alias = re.sub(r"[^A-Za-z0-9_-]", "_", tool_name).strip("_")[:64] or "agent_tool"
        aliases[tool_name] = tool_name
        aliases[alias] = tool_name
    return aliases


def _textual_tool_call_candidates(content: str) -> list[str]:
    candidates = [content]
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    for index, line in enumerate(lines):
        if "tool call" not in line.lower() and "工具调用" not in line:
            continue
        if index > 0 and "OfferMaster" in lines[index - 1]:
            candidates.append(f"{lines[index - 1]}\n{line}")
        if index + 1 < len(lines) and ("arguments" in lines[index + 1].lower() or "参数" in lines[index + 1]):
            candidates.append(f"{line}\n{lines[index + 1]}")
        candidates.append(line)
    return candidates


def _parse_tool_call_arguments(raw_arguments: str | None) -> dict[str, Any] | None:
    if raw_arguments is None or not str(raw_arguments).strip():
        return {}
    try:
        parsed = json.loads(raw_arguments)
    except Exception:
        return None
    return dict(parsed) if isinstance(parsed, dict) else None
