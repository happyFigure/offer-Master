from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.mcp_gateway.client import MCPToolCallResult


QQ_MAIL_READONLY_TOOL_NAMES = (
    "qq_mail.list_messages",
    "qq_mail.get_message",
)
_QQ_MAIL_TOOL_SET = frozenset(QQ_MAIL_READONLY_TOOL_NAMES)


def _normalized_name(tool_name: str) -> str:
    return str(tool_name or "").strip().lower().removeprefix("mcp.")


def qq_mail_tool_description(tool_name: str) -> str:
    return {
        "qq_mail.list_messages": "List recent QQ Mail messages without marking them as read. Read-only IMAPS access.",
        "qq_mail.get_message": "Read one QQ Mail message body by UID without marking it as read. Read-only IMAPS access.",
    }.get(_normalized_name(tool_name), f"Call the read-only QQ Mail MCP tool: {tool_name}.")


def qq_mail_tool_input_schema(tool_name: str) -> dict[str, Any] | None:
    schemas = {
        "qq_mail.list_messages": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                "since_days": {"type": "integer", "minimum": 1, "maximum": 365, "default": 30},
                "subject": {"type": ["string", "null"], "maxLength": 200},
                "sender": {"type": ["string", "null"], "maxLength": 200},
            },
            "additionalProperties": False,
        },
        "qq_mail.get_message": {
            "type": "object",
            "required": ["uid"],
            "properties": {
                "uid": {"type": "string", "pattern": "^[0-9]{1,32}$"},
                "include_body": {"type": "boolean", "default": True},
                "max_chars": {"type": "integer", "minimum": 1000, "maximum": 50000, "default": 12000},
            },
            "additionalProperties": False,
        },
    }
    schema = schemas.get(_normalized_name(tool_name))
    return deepcopy(schema) if schema is not None else None


def validate_qq_mail_tool_arguments(tool_name: str, arguments: dict[str, Any]) -> MCPToolCallResult | None:
    """Keep the QQ connector read-only and bound before starting IMAPS."""

    normalized = _normalized_name(tool_name)
    if not normalized.startswith("qq_mail."):
        return None
    if normalized not in _QQ_MAIL_TOOL_SET:
        return _policy_error(normalized, "Only list_messages and get_message are allowed for QQ Mail.")
    args = dict(arguments or {})
    if normalized == "qq_mail.list_messages":
        allowed = {"limit", "since_days", "subject", "sender"}
        if set(args) - allowed:
            return _policy_error(normalized, "Unknown QQ Mail list argument.")
        if not _bounded_int(args.get("limit", 20), 1, 100):
            return _policy_error(normalized, "limit must be between 1 and 100.")
        if not _bounded_int(args.get("since_days", 30), 1, 365):
            return _policy_error(normalized, "since_days must be between 1 and 365.")
        for key in ("subject", "sender"):
            if args.get(key) is not None and (not isinstance(args[key], str) or len(args[key]) > 200):
                return _policy_error(normalized, f"{key} must be a string of at most 200 characters.")
    else:
        allowed = {"uid", "include_body", "max_chars"}
        if set(args) - allowed:
            return _policy_error(normalized, "Unknown QQ Mail message argument.")
        uid = args.get("uid")
        if not isinstance(uid, str) or not uid.isdigit() or not 1 <= len(uid) <= 32:
            return _policy_error(normalized, "uid must be a numeric IMAP UID.")
        if not isinstance(args.get("include_body", True), bool):
            return _policy_error(normalized, "include_body must be boolean.")
        if not _bounded_int(args.get("max_chars", 12000), 1000, 50000):
            return _policy_error(normalized, "max_chars must be between 1000 and 50000.")
    return None


def _bounded_int(value: Any, minimum: int, maximum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum


def _policy_error(tool_name: str, message: str) -> MCPToolCallResult:
    return MCPToolCallResult(
        tool_name=tool_name,
        ok=False,
        error="QQ_MAIL_READ_ONLY_POLICY",
        result={"message": message, "read_only": True},
    )


__all__ = ["QQ_MAIL_READONLY_TOOL_NAMES", "qq_mail_tool_description", "qq_mail_tool_input_schema", "validate_qq_mail_tool_arguments"]
