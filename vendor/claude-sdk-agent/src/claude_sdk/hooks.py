from __future__ import annotations

import uuid
from typing import Any, Callable, Mapping

from ..hook_control import HookRuntimeRegistry

_HOOK_EVENTS = (
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "UserPromptSubmit",
    "Stop",
    "SubagentStop",
    "PreCompact",
    "Notification",
    "SubagentStart",
    "PermissionRequest",
)


def build_sdk_hooks(
    sdk_module: Any,
    *,
    frontend_session_id: str,
    registry: HookRuntimeRegistry | None,
    execution_resolver: Callable[[str], tuple[str, str]],
) -> dict[str, list[Any]]:
    if registry is None:
        return {}
    matcher_cls = getattr(sdk_module, "HookMatcher", None)
    if matcher_cls is None:
        return {}

    async def _callback(hook_input: Mapping[str, Any], tool_use_id: str | None, context: Mapping[str, Any]) -> Mapping[str, Any]:
        run_id, claude_session_id = execution_resolver(frontend_session_id)
        data = dict(hook_input or {})
        if context:
            data["context"] = dict(context)
        event_id = f"hookcb-{uuid.uuid4().hex}"
        await registry.record_event(
            session_id=frontend_session_id,
            run_id=run_id,
            claude_session_id=claude_session_id or str(hook_input.get("session_id") or "").strip(),
            event_id=event_id,
            hook_event_name=str(hook_input.get("hook_event_name") or "").strip(),
            phase="callback",
            source="sdk_callback",
            status="completed",
            tool_name=str(hook_input.get("tool_name") or "").strip(),
            tool_use_id=str(tool_use_id or hook_input.get("tool_use_id") or "").strip(),
            agent_id=str(hook_input.get("agent_id") or "").strip(),
            agent_type=str(hook_input.get("agent_type") or "").strip(),
            title=str(hook_input.get("title") or "").strip(),
            notification_type=str(hook_input.get("notification_type") or "").strip(),
            data=data,
            output={"continue": True},
        )
        return {"continue_": True}

    matcher = matcher_cls(matcher=None, hooks=[_callback], timeout=60.0)
    return {event_name: [matcher] for event_name in _HOOK_EVENTS}


def hook_stream_payload(item: Any) -> Mapping[str, Any] | None:
    attachment = item.get("attachment") if isinstance(item, Mapping) else getattr(item, "attachment", None)
    if isinstance(attachment, Mapping):
        payload = _hook_attachment_payload(item, attachment)
        if payload is not None:
            return payload
    if type(item).__name__ != "HookEventMessage":
        return None
    data = getattr(item, "data", None)
    if not isinstance(data, Mapping):
        data = {}
    phase = str(getattr(item, "subtype", "") or "").strip() or "hook_event"
    event_id = str(data.get("callback_id") or getattr(item, "uuid", "") or "").strip() or f"hookmsg-{uuid.uuid4().hex}"
    status = "running" if phase == "hook_started" else "completed"
    return {
        "eventId": event_id,
        "claudeSessionId": str(getattr(item, "session_id", "") or data.get("session_id") or "").strip(),
        "hookEventName": str(getattr(item, "hook_event_name", "") or data.get("hook_event_name") or "").strip(),
        "phase": phase,
        "source": "sdk_stream",
        "status": status,
        "matcher": str(data.get("matcher") or "").strip(),
        "toolName": str(data.get("tool_name") or "").strip(),
        "toolUseId": str(data.get("tool_use_id") or "").strip(),
        "agentId": str(data.get("agent_id") or "").strip(),
        "agentType": str(data.get("agent_type") or "").strip(),
        "title": str(data.get("title") or data.get("message") or "").strip(),
        "notificationType": str(data.get("notification_type") or "").strip(),
        "data": dict(data),
        "output": dict(data.get("output")) if isinstance(data.get("output"), Mapping) else {},
        "outcome": str(data.get("outcome") or "").strip(),
        "exitCode": _as_int_or_none(data.get("exit_code")),
    }


def _as_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _hook_attachment_payload(item: Any, attachment: Mapping[str, Any]) -> Mapping[str, Any] | None:
    hook_event_name = str(
        attachment.get("hookEvent")
        or attachment.get("hook_event_name")
        or attachment.get("hookName")
        or ""
    ).strip()
    hook_name = str(attachment.get("hookName") or "").strip()
    if not hook_event_name and not hook_name:
        return None
    event_id = str(
        attachment.get("toolUseID")
        or attachment.get("toolUseId")
        or attachment.get("tool_use_id")
        or getattr(item, "uuid", "")
        or ""
    ).strip() or f"hookmsg-{uuid.uuid4().hex}"
    attachment_type = str(attachment.get("type") or "").strip() or "hook_attachment"
    exit_code = _as_int_or_none(attachment.get("exitCode") or attachment.get("exit_code"))
    status = "failed" if attachment_type == "hook_non_blocking_error" or (exit_code is not None and exit_code != 0) else "completed"
    return {
        "eventId": event_id,
        "claudeSessionId": str(
            attachment.get("session_id")
            or attachment.get("sessionId")
            or getattr(item, "session_id", "")
            or ""
        ).strip(),
        "hookEventName": hook_event_name or hook_name,
        "phase": attachment_type,
        "source": "sdk_attachment",
        "status": status,
        "matcher": str(attachment.get("matcher") or "").strip(),
        "toolName": str(attachment.get("toolName") or attachment.get("tool_name") or "").strip(),
        "toolUseId": event_id,
        "agentId": str(attachment.get("agentId") or attachment.get("agent_id") or "").strip(),
        "agentType": str(attachment.get("agentType") or attachment.get("agent_type") or "").strip(),
        "title": hook_name or hook_event_name,
        "notificationType": attachment_type,
        "data": dict(attachment),
        "output": dict(attachment.get("output")) if isinstance(attachment.get("output"), Mapping) else {},
        "outcome": str(attachment.get("stderr") or attachment.get("outcome") or "").strip(),
        "exitCode": exit_code,
    }
