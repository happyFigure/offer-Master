from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.agent_runtime.state import AgentState


@dataclass(frozen=True)
class ToolObservation:
    """Normalized evidence returned by a tool or skill.

    Executors can return different payload shapes. The final-answer layer should
    not care about each executor's private envelope; it only needs a compact,
    auditable observation that can be fed back to the main agent.
    """

    tool_name: str
    status: str
    ok: bool
    operation: str | None = None
    evidence: str = ""
    error: str | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "ok": self.ok,
            "operation": self.operation,
            "error": self.error,
            "evidence": self.evidence,
        }


def extract_tool_observations(state: AgentState) -> list[ToolObservation]:
    """Extract tool observations from the runtime transcript in AgentState."""

    observations: list[ToolObservation] = []
    for message in state.llm_messages:
        payload = _content_json_from_message(message)
        if not _is_tool_result_payload(payload):
            continue
        observations.append(_observation_from_payload(payload))
    return observations


def _content_json_from_message(message: Any) -> dict[str, Any] | None:
    if not isinstance(message, dict):
        return None

    metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
    content_json = metadata.get("content_json")
    if isinstance(content_json, dict):
        return content_json

    content = str(message.get("content") or "")
    if "\n" not in content:
        return None
    _prefix, json_text = content.split("\n", 1)
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _is_tool_result_payload(payload: dict[str, Any] | None) -> bool:
    return isinstance(payload, dict) and bool(payload.get("tool_name")) and "status" in payload


def _observation_from_payload(payload: dict[str, Any]) -> ToolObservation:
    tool_name = str(payload.get("tool_name") or "")
    status = str(payload.get("status") or "unknown")
    result_payload = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    operation = _operation_from_payload(payload, result_payload)
    error = _error_from_payload(payload, result_payload)
    ok = status == "succeeded" and _result_ok(result_payload)
    return ToolObservation(
        tool_name=tool_name,
        status=status,
        ok=ok,
        operation=operation,
        evidence=_evidence_from_payload(payload, result_payload),
        error=error,
        raw_payload=dict(payload),
    )


def _operation_from_payload(payload: dict[str, Any], result_payload: dict[str, Any]) -> str | None:
    for candidate in (
        result_payload.get("operation"),
        payload.get("operation"),
        _nested_dict(result_payload, "result").get("operation"),
        _nested_dict(_nested_dict(result_payload, "result"), "result").get("operation"),
    ):
        text = str(candidate or "").strip()
        if text:
            return text
    return None


def _error_from_payload(payload: dict[str, Any], result_payload: dict[str, Any]) -> str | None:
    for candidate in (
        payload.get("error"),
        result_payload.get("error"),
        _nested_dict(result_payload, "result").get("error"),
        _nested_dict(_nested_dict(result_payload, "result"), "result").get("error"),
    ):
        text = str(candidate or "").strip()
        if text:
            return text
    return None


def _result_ok(result_payload: dict[str, Any]) -> bool:
    if "ok" in result_payload:
        return bool(result_payload.get("ok"))
    nested = result_payload.get("result")
    if isinstance(nested, dict) and "ok" in nested:
        return bool(nested.get("ok"))
    return True


def _evidence_from_payload(payload: dict[str, Any], result_payload: dict[str, Any]) -> str:
    evidence_payload = {
        "summary": result_payload.get("summary") or payload.get("summary"),
        "operation": _operation_from_payload(payload, result_payload),
        "result": result_payload.get("result", result_payload),
        "error": _error_from_payload(payload, result_payload),
    }
    return _json_dumps_compact(_compact_value(evidence_payload))


def _nested_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    nested = payload.get(key)
    return nested if isinstance(nested, dict) else {}


def _compact_value(value: Any, *, string_limit: int = 12000, list_limit: int = 20) -> Any:
    if isinstance(value, str):
        return value if len(value) <= string_limit else f"{value[:string_limit]}\n\n[内容过长，已截断用于最终回答生成。]"
    if isinstance(value, dict):
        return {str(key): _compact_value(item, string_limit=string_limit, list_limit=list_limit) for key, item in value.items()}
    if isinstance(value, list):
        compact_items = [_compact_value(item, string_limit=string_limit, list_limit=list_limit) for item in value[:list_limit]]
        if len(value) > list_limit:
            compact_items.append(f"[列表过长，已截断 {len(value) - list_limit} 项。]")
        return compact_items
    return value


def _json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
