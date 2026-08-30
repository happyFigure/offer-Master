from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_PROTOCOL_BLOCK_RE = re.compile(
    r"(?is)\[(?:tool|toolcall|tool_call|approval|run_result|observation)\].*?\[/(?:tool|toolcall|tool_call|approval|run_result|observation)\]"
)
_FENCED_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_-]*\n(?P<body>.*?)```", re.DOTALL)
_PROTOCOL_LINE_RE = re.compile(
    r"^\s*(?:Tool call|Tool result|TOOL_CALL|TOOL_RESULT|工具调用|工具结果)\s*[:：].*$",
    re.IGNORECASE,
)
_ASSISTANT_LABEL_RE = re.compile(r"^\s*(?:\*\*)?OfferMaster\s+AI(?:\*\*)?\s*[:：]?$", re.IGNORECASE)
_INTERNAL_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_REMOVED_MARKER = "__OFFERMASTER_INTERNAL_PROTOCOL_REMOVED__"
_FALSE_TOOL_EXECUTION_CLAIM_RE = re.compile(
    r"(?i)(?:我|本轮|这轮|系统|主\s*agent|agent)?\s*"
    r"(?:已经|已)\s*(?:调用|使用|执行|运行)(?:了)?\s*"
    r"(?:联网搜索|网页搜索|公开搜索|工具|能力|external\.web_search|filesystem\.[a-z0-9_]+|[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*)"
)
_FALSE_TOOL_EXECUTION_CLAIM_PREFIXES = (
    "我已",
    "我已经",
    "本轮已",
    "本轮已经",
    "这轮已",
    "这轮已经",
    "系统已",
    "系统已经",
    "主agent已",
    "主agent已经",
    "agent已",
    "agent已经",
    "已经",
    "已",
)


@dataclass(frozen=True)
class SanitizedOutput:
    content: str
    removed_internal_protocol: bool
    needs_regeneration: bool
    removed_fragments: tuple[str, ...] = ()


def sanitize_agent_final_answer(content: str) -> SanitizedOutput:
    """Remove internal tool protocol text that should never be shown to users."""

    original = str(content or "").replace("\r\n", "\n")
    removed: list[str] = []

    def remove_protocol_block(match: re.Match[str]) -> str:
        removed.append(match.group(0).strip())
        return f"\n{_REMOVED_MARKER}\n"

    cleaned = _PROTOCOL_BLOCK_RE.sub(remove_protocol_block, original)

    def remove_internal_fence(match: re.Match[str]) -> str:
        body = match.group("body")
        if _looks_like_internal_protocol(body):
            removed.append(match.group(0).strip())
            return f"\n{_REMOVED_MARKER}\n"
        return match.group(0)

    cleaned = _FENCED_BLOCK_RE.sub(remove_internal_fence, cleaned)

    kept_lines: list[str] = []
    skip_blank_after_removed_protocol = False
    for line in cleaned.split("\n"):
        stripped = line.strip()
        if stripped == _REMOVED_MARKER or _looks_like_internal_protocol_line(line):
            if stripped != _REMOVED_MARKER:
                removed.append(stripped)
            while kept_lines and not kept_lines[-1].strip():
                kept_lines.pop()
            skip_blank_after_removed_protocol = True
            continue
        if skip_blank_after_removed_protocol and not stripped:
            continue
        skip_blank_after_removed_protocol = False
        kept_lines.append(line)

    final_content = _compact_blank_lines("\n".join(kept_lines)).strip()
    removed_internal_protocol = bool(removed)
    if removed_internal_protocol and _is_only_assistant_shell(final_content):
        final_content = ""
    return SanitizedOutput(
        content=final_content,
        removed_internal_protocol=removed_internal_protocol,
        needs_regeneration=removed_internal_protocol and not final_content,
        removed_fragments=tuple(fragment for fragment in removed if fragment),
    )


def contains_false_tool_execution_claim(content: str) -> bool:
    return _FALSE_TOOL_EXECUTION_CLAIM_RE.search(str(content or "")) is not None


def could_be_false_tool_execution_claim_prefix(content: str) -> bool:
    normalized = re.sub(r"\s+", "", str(content or "").replace("\r\n", "\n").lstrip()).lower()
    if not normalized:
        return True
    return any(prefix.lower().startswith(normalized) for prefix in _FALSE_TOOL_EXECUTION_CLAIM_PREFIXES)


def false_tool_execution_claim_fallback_response() -> str:
    return "本轮没有真实工具执行记录，所以不能声称已经调用了工具。请重新发送问题，我会重新进入工具链真实执行。"


def _looks_like_internal_protocol_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _PROTOCOL_LINE_RE.match(stripped):
        return True
    if stripped.startswith("{") and stripped.endswith("}"):
        return _looks_like_internal_protocol(stripped)
    return False


def _looks_like_internal_protocol(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if lowered.startswith("tool call:") or lowered.startswith("tool result:"):
        return True
    if "[tool" in lowered or "[/tool" in lowered or "[run_result]" in lowered:
        return True
    parsed = _parse_json(stripped)
    if parsed is not None:
        return _json_looks_like_tool_protocol(parsed)
    protocol_markers = ("tool_call", "tool_calls", "tool_name", "run_result")
    return any(marker in lowered for marker in protocol_markers) and ("arguments" in lowered or "result" in lowered)


def _json_looks_like_tool_protocol(value: Any) -> bool:
    if isinstance(value, list):
        return any(_json_looks_like_tool_protocol(item) for item in value)
    if not isinstance(value, dict):
        return False
    keys = {str(key).lower() for key in value.keys()}
    if keys.intersection({"tool_call", "tool_calls", "tool_name", "run_result", "tool_result"}):
        return True
    name = value.get("name") or value.get("tool")
    if isinstance(name, str) and _INTERNAL_TOOL_NAME_RE.match(name) and keys.intersection({"arguments", "input"}):
        return True
    return any(_json_looks_like_tool_protocol(item) for item in value.values())


def _parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _compact_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)


def _is_only_assistant_shell(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    lines = [line.strip() for line in stripped.split("\n") if line.strip()]
    return bool(lines) and all(_ASSISTANT_LABEL_RE.match(line) for line in lines)
