from __future__ import annotations

import json
from typing import Any


def event_to_text_delta(item: Any) -> str:
    event = getattr(item, "event", None)
    if event is None and isinstance(item, dict):
        event = item.get("event")
    if event is None:
        return ""
    event_type = _field(event, "type")
    if event_type != "content_block_delta":
        return ""
    delta = _field(event, "delta")
    if delta is None:
        return ""
    delta_type = _field(delta, "type")
    if delta_type not in {"text_delta", "output_text_delta"}:
        return ""
    return str(_field(delta, "text") or "")


def stream_event_tool_start(item: Any) -> dict[str, Any] | None:
    event = getattr(item, "event", None)
    if event is None and isinstance(item, dict):
        event = item.get("event")
    if not isinstance(event, dict):
        return None
    if str(event.get("type") or "").strip() != "content_block_start":
        return None
    block = event.get("content_block")
    block_type = str(_field(block, "type") or "").strip()
    if block_type not in {"tool_use", "server_tool_use"}:
        return None
    tool_call_id = str(_field(block, "id") or "").strip()
    name = str(_field(block, "name") or "").strip()
    arguments = _field(block, "input")
    return {
        "toolCallId": tool_call_id,
        "name": name,
        "arguments": arguments if isinstance(arguments, dict) else {},
        "toolType": "server_tool" if block_type == "server_tool_use" else "tool",
    }


def content_tool_starts(item: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in _content_blocks(item):
        block_type = str(_field(block, "type") or _block_type_name(block) or "").strip()
        if block_type not in {"tool_use", "server_tool_use"}:
            continue
        tool_call_id = str(_field(block, "id") or "").strip()
        name = str(_field(block, "name") or "").strip()
        arguments = _field(block, "input")
        events.append(
            {
                "toolCallId": tool_call_id,
                "name": name,
                "arguments": arguments if isinstance(arguments, dict) else {},
                "toolType": "server_tool" if block_type == "server_tool_use" else "tool",
            }
        )
    return events


def item_tool_file_paths(item: Any) -> list[str]:
    paths: list[str] = []
    for event in content_tool_starts(item):
        name = str(event.get("name") or "").strip().lower()
        arguments = event.get("arguments")
        if not isinstance(arguments, dict):
            continue
        path = _tool_file_path(name, arguments)
        if path and path not in paths:
            paths.append(path)
    return paths


def _tool_file_path(name: str, arguments: dict[str, Any]) -> str:
    if name in {"write", "edit", "multiedit", "notebookedit"}:
        return str(arguments.get("file_path") or arguments.get("path") or arguments.get("notebook_path") or "").strip()
    return ""


def content_tool_results(item: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in _content_blocks(item):
        block_type = str(_field(block, "type") or _block_type_name(block) or "").strip()
        if block_type not in {"tool_result", "advisor_tool_result", "server_tool_result"}:
            continue
        tool_call_id = str(_field(block, "tool_use_id") or "").strip()
        content = _field(block, "content")
        is_error = bool(_field(block, "is_error"))
        events.append(
            {
                "toolCallId": tool_call_id,
                "status": "failed" if is_error else "completed",
                "result": stringify_tool_content(content),
            }
        )
    return events


def task_message_payload(item: Any) -> dict[str, Any] | None:
    kind = type(item).__name__
    if kind == "TaskStartedMessage":
        return {
            "phase": "start",
            "taskId": str(_field(item, "task_id") or "").strip(),
            "taskType": str(_field(item, "task_type") or "").strip(),
            "toolCallId": str(_field(item, "tool_use_id") or "").strip(),
            "name": str(_field(item, "description") or _field(item, "task_type") or "").strip(),
            "status": "running",
            "log": str(_field(item, "description") or "").strip(),
        }
    if kind == "TaskProgressMessage":
        return {
            "phase": "update",
            "taskId": str(_field(item, "task_id") or "").strip(),
            "taskType": str(_field(item, "task_type") or "").strip(),
            "toolCallId": str(_field(item, "tool_use_id") or "").strip(),
            "name": str(_field(item, "description") or _field(item, "last_tool_name") or "").strip(),
            "status": "running",
            "log": str(_field(item, "description") or "").strip(),
        }
    if kind == "TaskNotificationMessage":
        summary = str(_field(item, "summary") or "").strip()
        status = _normalize_task_status(_field(item, "status"), summary, default="completed")
        return {
            "phase": "end",
            "taskId": str(_field(item, "task_id") or "").strip(),
            "taskType": str(_field(item, "task_type") or "").strip(),
            "toolCallId": str(_field(item, "tool_use_id") or "").strip(),
            "name": str(_field(item, "description") or _field(item, "task_type") or "").strip(),
            "status": status,
            "log": summary,
            "result": summary,
        }
    if kind == "TaskUpdatedMessage":
        patch = _field(item, "patch")
        patch_status = patch.get("status") if isinstance(patch, dict) else None
        patch_text = stringify_tool_content(patch)
        status = _normalize_task_status(patch_status or _field(item, "status"), patch_text, default="running")
        terminal_statuses = {"completed", "failed", "failure", "error", "killed", "stopped", "cancelled", "canceled"}
        return {
            "phase": "end" if status.lower() in terminal_statuses else "update",
            "taskId": str(_field(item, "task_id") or _field(patch, "task_id") or _field(patch, "taskId") or "").strip(),
            "taskType": str(_field(item, "task_type") or _field(patch, "task_type") or _field(patch, "taskType") or "").strip(),
            "toolCallId": str(
                _field(item, "tool_use_id")
                or _field(patch, "tool_use_id")
                or _field(patch, "toolUseId")
                or _field(patch, "tool_call_id")
                or _field(patch, "toolCallId")
                or ""
            ).strip(),
            "name": str(
                _field(item, "description")
                or _field(patch, "description")
                or _field(patch, "name")
                or _field(patch, "task_type")
                or _field(patch, "taskType")
                or ""
            ).strip(),
            "status": status or "running",
            "log": patch_text,
            "result": patch_text,
        }
    return None


def _normalize_task_status(value: Any, text: str, *, default: str) -> str:
    del text
    status = str(value or "").strip().lower()
    if status in {"failed", "failure", "error", "errored"}:
        return "failed"
    if status in {"cancelled", "canceled", "cancel_requested"}:
        return "cancelled"
    if status in {"killed", "stopped"}:
        return status
    if status in {"completed", "complete", "done", "success", "succeeded"}:
        return "completed"
    if status:
        return status
    return default


def extract_session_id(item: Any) -> str:
    for container in (item, getattr(item, "result", None), getattr(item, "message", None)):
        value = _field(container, "session_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    result = getattr(item, "result", None)
    for container in (result, _field(result, "message"), _field(result, "metadata"), _field(item, "metadata")):
        value = _field(container, "session_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def assistant_text(item: Any) -> str:
    text = str(_field(item, "text") or "").strip()
    if text:
        return text
    direct_content = _field(item, "content")
    direct_text = _content_text(direct_content)
    if direct_text:
        return direct_text
    message = getattr(item, "message", None)
    return _content_text(_field(message, "content"))


def final_result_error_text(item: Any) -> str:
    if not is_final_result(item):
        return ""
    subtype = str(_field(item, "subtype") or _field(getattr(item, "result", None), "subtype") or "").strip()
    is_error = bool(_field(item, "is_error") or _field(getattr(item, "result", None), "is_error"))
    if not is_error and not subtype.lower().startswith("error"):
        return ""
    if subtype == "error_max_turns":
        return "外部执行器已达到最大回合数，未能生成最终回复。请缩小任务范围或提高 max_turns 后重试。"
    if subtype:
        return f"外部执行器未能生成最终回复：{subtype}。"
    return "外部执行器未能生成最终回复。"


def is_final_result(item: Any) -> bool:
    return type(item).__name__ == "ResultMessage" or bool(getattr(item, "result", None) is not None)


def is_assistant_message(item: Any) -> bool:
    return type(item).__name__ == "AssistantMessage"


def is_compact_boundary(item: Any) -> bool:
    return type(item).__name__ == "SystemMessage" and str(_field(item, "subtype") or "").strip() == "compact_boundary"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        block_type = _field(block, "type")
        text = str(_field(block, "text") or "").strip()
        if text and block_type in (None, "", "text", "output_text"):
            parts.append(text)
            continue
        if block_type not in {"text", "output_text"}:
            continue
    return "\n".join(parts).strip()


def _content_blocks(item: Any) -> list[Any]:
    content = _field(item, "content")
    if isinstance(content, list):
        return content
    message = getattr(item, "message", None)
    content = _field(message, "content")
    if isinstance(content, list):
        return content
    return []


def stringify_tool_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = str(_field(block, "text") or "").strip()
            if text:
                parts.append(text)
                continue
            if isinstance(block, dict):
                rendered = stringify_tool_content(block)
                if rendered:
                    parts.append(rendered)
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        text = str(content.get("text") or "").strip()
        if text:
            return text
        nested = content.get("content")
        if nested is not None:
            rendered = stringify_tool_content(nested)
            if rendered:
                return rendered
        try:
            return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(content).strip()
    return str(content).strip()


def _block_type_name(block: Any) -> str:
    name = type(block).__name__
    mapping = {
        "ToolUseBlock": "tool_use",
        "ToolResultBlock": "tool_result",
        "ServerToolUseBlock": "server_tool_use",
        "ServerToolResultBlock": "server_tool_result",
    }
    return mapping.get(name, "")


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
