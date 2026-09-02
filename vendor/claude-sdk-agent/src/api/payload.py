from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


HISTORY_CONTEXT_MARKER = "[Chat messages since your last reply - for context]"
CURRENT_MESSAGE_MARKER = "[Current message - respond to this]"
RUNTIME_RETRY_NOTE_MARKER = "[runtime-retry]"
PROTOCOL_SHELL_MARKERS = (
    "[tool] ",
    "[task] ",
    "[approval] ",
    "[question] ",
    "[command] ",
    "[meta] ",
    "[artifacts] ",
)
CLAUDE_CODE_RUNTIME_COMMAND_SOURCE = "claude-code"
CLAUDE_CODE_RUNTIME_COMMANDS = {
    "compact": "/compact",
    "context": "/context",
    "usage": "/usage",
    "goal": "/goal",
}
_CLAUDE_CODE_DYNAMIC_COMMAND_PATTERN = re.compile(r"^/[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True, slots=True)
class RuntimeCommand:
    source: str
    command_id: str
    command: str
    args: Mapping[str, Any]
    display_name: str
    request_id: str
    kind: str = ""

    @property
    def prompt_text(self) -> str:
        argument_text = str(self.args.get("text") or "").strip()
        return f"{self.command} {argument_text}".strip()

    def shell_payload(
        self,
        *,
        status: str,
        phase: str,
        message: str = "",
        error: str = "",
        result: str = "",
    ) -> dict[str, Any]:
        return {
            "source": self.source,
            "commandId": self.command_id,
            "command": self.command,
            "args": dict(self.args),
            "displayName": self.display_name,
            "requestId": self.request_id,
            "status": status,
            "phase": phase,
            **({"kind": self.kind} if self.kind else {}),
            **({"message": message} if message else {}),
            **({"error": error} if error else {}),
            **({"result": result} if result else {}),
        }


def normalize_session_id_in_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, dict):
        return
    sid = payload.get("session_id") or payload.get("sessionId")
    if isinstance(sid, str) and sid.strip():
        payload["session_id"] = sid.strip()
        return
    if sid is not None and isinstance(sid, (int, float)) and not isinstance(sid, bool):
        payload["session_id"] = str(int(sid)) if isinstance(sid, float) and sid == int(sid) else str(sid)
        return
    for key in ("user", "user_id", "userId"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            payload["session_id"] = value.strip()
            return
        if value is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
            payload["session_id"] = str(int(value)) if isinstance(value, float) and value == int(value) else str(value)
            return
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("session_id", "sessionId", "user", "user_id", "userId"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                payload["session_id"] = value.strip()
                return
            if value is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
                payload["session_id"] = str(int(value)) if isinstance(value, float) and value == int(value) else str(value)
                return


def extract_text_from_content_parts(content: Any) -> str:
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes, bytearray)):
        return ""
    out: list[str] = []
    for item in content:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") != "text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text:
            out.append(text)
    return "\n\n".join(out).strip()


def sanitize_incoming_assistant_content(text: str) -> str:
    if not text:
        return ""
    cleaned = "\n".join(
        line
        for line in text.splitlines()
        if not line.lstrip().startswith(RUNTIME_RETRY_NOTE_MARKER)
        and not any(line.lstrip().startswith(marker) for marker in PROTOCOL_SHELL_MARKERS)
    )
    return cleaned.strip()


def extract_latest_user_text(payload: Mapping[str, Any]) -> str:
    if isinstance(payload.get("input"), str):
        return str(payload.get("input") or "")
    if isinstance(payload.get("message"), str):
        return str(payload.get("message") or "")
    messages = payload.get("messages") or []
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, Mapping) or str(message.get("role") or "").strip().lower() != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content
            text = extract_text_from_content_parts(content)
            if text:
                return text
    return ""


def extract_session_id(payload: Mapping[str, Any]) -> str | None:
    for key in ("session_id", "sessionId", "user", "user_id", "userId"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(int(value)) if isinstance(value, float) and value == int(value) else str(value)
    return None


def extract_runtime_command(payload: Mapping[str, Any]) -> RuntimeCommand | None:
    raw = payload.get("runtime_command") or payload.get("runtimeCommand")
    metadata = payload.get("metadata")
    if not isinstance(raw, Mapping) and isinstance(metadata, Mapping):
        raw = metadata.get("runtimeCommand") or metadata.get("runtime_command")
    if not isinstance(raw, Mapping):
        return _extract_runtime_command_from_latest_user_text(payload)

    source = str(raw.get("source") or "").strip()
    command_id = str(raw.get("commandId") or raw.get("command_id") or "").strip().lower()
    command = str(raw.get("command") or "").strip().lower()
    if not command_id and command.startswith("/"):
        command_id = command[1:]
    if source != CLAUDE_CODE_RUNTIME_COMMAND_SOURCE:
        return None
    expected_command = CLAUDE_CODE_RUNTIME_COMMANDS.get(command_id)
    dynamic_kind = str(raw.get("kind") or raw.get("resourceKind") or raw.get("resource_kind") or "").strip().lower()
    if expected_command is None and dynamic_kind not in {"command", "skill", "workflow"}:
        return None
    resolved_command = expected_command or command or (f"/{command_id}" if command_id else "")
    if not _CLAUDE_CODE_DYNAMIC_COMMAND_PATTERN.fullmatch(resolved_command):
        return None
    resolved_command_id = resolved_command[1:].lower()
    if command_id and command_id != resolved_command_id:
        return None
    command_id = resolved_command_id
    if expected_command and command and command != expected_command:
        return None

    args = raw.get("args")
    normalized_args = dict(args) if isinstance(args, Mapping) else {}
    argument_text = normalized_args.get("text")
    if argument_text is not None and not isinstance(argument_text, str):
        return None
    if normalized_args.get("text") and expected_command and command_id != "goal":
        return None

    display_name = str(raw.get("displayName") or raw.get("display_name") or resolved_command).strip()
    request_id = str(raw.get("requestId") or raw.get("request_id") or "").strip()
    return RuntimeCommand(
        source=source,
        command_id=command_id,
        command=resolved_command,
        args=normalized_args,
        display_name=display_name,
        request_id=request_id,
        kind=dynamic_kind,
    )


def _extract_runtime_command_from_latest_user_text(payload: Mapping[str, Any]) -> RuntimeCommand | None:
    text = extract_latest_user_text(payload).strip()
    if not text or "\n" in text:
        return None
    parts = text.split(maxsplit=1)
    command = parts[0].strip().lower()
    if not command.startswith("/"):
        return None
    command_id = command[1:]
    expected_command = CLAUDE_CODE_RUNTIME_COMMANDS.get(command_id)
    if not expected_command or command != expected_command:
        return None
    argument_text = parts[1].strip() if len(parts) > 1 else ""
    if argument_text and command_id != "goal":
        return None
    return RuntimeCommand(
        source=CLAUDE_CODE_RUNTIME_COMMAND_SOURCE,
        command_id=command_id,
        command=expected_command,
        args={"text": argument_text} if argument_text else {},
        display_name=expected_command,
        request_id=f"cmd-{command_id}-fallback",
    )


def _metadata_agentconfig(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    cfg = metadata.get("agentconfig") or metadata.get("agentConfig")
    return cfg if isinstance(cfg, Mapping) else None


def extract_artifacts_enabled(payload: Mapping[str, Any]) -> bool:
    cfg = _metadata_agentconfig(payload)
    if not isinstance(cfg, Mapping):
        return False
    raw = cfg.get("artifacts_enabled")
    if raw is None:
        raw = cfg.get("artifactsEnabled")
    if raw is None:
        raw = cfg.get("artifacts")
        if isinstance(raw, Mapping):
            raw = raw.get("enabled")
    return _coerce_bool(raw, default=False)


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return default


def metadata_userinfo_system_block(payload: Mapping[str, Any]) -> str:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return ""
    userinfo = metadata.get("userinfo")
    if not isinstance(userinfo, Mapping):
        return ""
    name = ""
    employ_num = ""
    for key in ("name", "userName", "user_name"):
        value = str(userinfo.get(key) or "").strip()
        if value:
            name = value
            break
    for key in ("employNum", "employ_num", "employeeNum", "employee_num", "empId", "emp_id"):
        value = str(userinfo.get(key) or "").strip()
        if value:
            employ_num = value
            break
    parts = [value for value in (name, employ_num) if value]
    return "当前的用户是： " + " ".join(parts) if parts else ""


def extract_agentprompt(payload: Mapping[str, Any]) -> str:
    cfg = _metadata_agentconfig(payload)
    if not isinstance(cfg, Mapping):
        return ""
    raw = cfg.get("prompt")
    return raw.strip() if isinstance(raw, str) else ""


def build_initial_prompt(payload: Mapping[str, Any]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return extract_latest_user_text(payload)
    history_lines: list[str] = []
    current_text = ""
    user_indexes = [idx for idx, item in enumerate(messages) if isinstance(item, Mapping) and str(item.get("role") or "").strip().lower() == "user"]
    last_user_index = user_indexes[-1] if user_indexes else len(messages) - 1
    for idx, item in enumerate(messages):
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "").strip().lower() or "user"
        content = item.get("content")
        if isinstance(content, str):
            text = content.strip()
        else:
            text = extract_text_from_content_parts(content)
        if role == "assistant":
            text = sanitize_incoming_assistant_content(text)
        if not text:
            continue
        if idx == last_user_index and role == "user":
            current_text = text
            continue
        history_lines.append(f"{role}: {text}")
    if not history_lines:
        return current_text
    history = "\n\n".join(history_lines).strip()
    return f"{HISTORY_CONTEXT_MARKER}\n{history}\n\n{CURRENT_MESSAGE_MARKER}\n{current_text}".strip()


def build_system_prompt(payload: Mapping[str, Any]) -> str:
    blocks = [value for value in (metadata_userinfo_system_block(payload), extract_agentprompt(payload)) if value]
    return "\n\n".join(blocks).strip()
