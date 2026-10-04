from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PENDING_OPERATION_METADATA_KEY = "pending_operation"
PENDING_OPERATION_SOURCE = "pending_operation_v1"


@dataclass(frozen=True)
class PendingOperationFrame:
    """A known user operation that cannot run yet because some arguments are missing.

    `active_file` only answers "which file are we talking about". This frame
    answers "what unfinished operation are we still trying to complete" so a
    terse follow-up like "你自己起名字" can resume the old copy task.
    """

    capability: str
    operation: str
    known_args: dict[str, Any] = field(default_factory=dict)
    missing_args: tuple[str, ...] = ()
    constraints: dict[str, Any] = field(default_factory=dict)
    resume_hints: tuple[str, ...] = ()
    source: str = PENDING_OPERATION_SOURCE

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "capability": self.capability,
            "operation": self.operation,
            "known_args": dict(self.known_args),
            "missing_args": list(self.missing_args),
            "constraints": dict(self.constraints),
            "resume_hints": list(self.resume_hints),
        }


def pending_operation_from_metadata(metadata: dict[str, Any] | None) -> PendingOperationFrame | None:
    if not isinstance(metadata, dict):
        return None
    payload = metadata.get(PENDING_OPERATION_METADATA_KEY)
    if not isinstance(payload, dict):
        return None
    capability = str(payload.get("capability") or "").strip()
    operation = str(payload.get("operation") or "").strip()
    if not capability or not operation:
        return None
    return PendingOperationFrame(
        capability=capability,
        operation=operation,
        known_args=_dict_or_empty(payload.get("known_args")),
        missing_args=_tuple_of_strings(payload.get("missing_args")),
        constraints=_dict_or_empty(payload.get("constraints")),
        resume_hints=_tuple_of_strings(payload.get("resume_hints")),
        source=str(payload.get("source") or PENDING_OPERATION_SOURCE),
    )


def with_pending_operation(metadata: dict[str, Any] | None, frame: PendingOperationFrame | None) -> dict[str, Any]:
    next_metadata = dict(metadata or {})
    if frame is None:
        return clear_pending_operation(next_metadata)
    next_metadata[PENDING_OPERATION_METADATA_KEY] = frame.to_metadata_dict()
    return next_metadata


def clear_pending_operation(metadata: dict[str, Any] | None) -> dict[str, Any]:
    next_metadata = dict(metadata or {})
    next_metadata.pop(PENDING_OPERATION_METADATA_KEY, None)
    return next_metadata


def build_pending_operation_from_tool_input_completion(
    *,
    tool_name: str,
    completion: Any,
    context_metadata: dict[str, Any] | None,
    user_message: str,
) -> PendingOperationFrame | None:
    missing_args = _tuple_of_strings(getattr(completion, "missing_required_fields", ()))
    if not missing_args:
        return None

    metadata = context_metadata if isinstance(context_metadata, dict) else {}
    operation = _operation_from_tool_name(tool_name=tool_name, metadata=metadata, completion=completion)
    if not operation:
        return None

    known_args = _dict_or_empty(getattr(completion, "tool_input", {}))
    if operation in {"copy_file", "move_file", "rename_file"} and not str(known_args.get("src") or "").strip():
        active_file = metadata.get("active_file") if isinstance(metadata.get("active_file"), dict) else {}
        active_path = str(active_file.get("path") or "").strip()
        if active_path:
            known_args["src"] = active_path

    constraints = {}
    operation_intent = known_args.get("operation_intent")
    if isinstance(operation_intent, dict):
        constraints["operation_intent"] = dict(operation_intent)
    capability = "skill.filesystem" if operation else str(tool_name or "").strip()
    return PendingOperationFrame(
        capability=capability,
        operation=operation,
        known_args=known_args,
        missing_args=missing_args,
        constraints=constraints,
        resume_hints=_resume_hints_for(operation=operation, missing_args=missing_args, user_message=user_message),
    )


def _operation_from_tool_name(*, tool_name: str, metadata: dict[str, Any], completion: Any) -> str:
    tool = str(tool_name or "").strip()
    if tool == "skill.filesystem":
        tool_input = _dict_or_empty(getattr(completion, "tool_input", {}))
        return str(
            tool_input.get("operation")
            or metadata.get("filesystem_operation")
            or ""
        ).strip()
    if tool.startswith("filesystem."):
        return tool.split(".", 1)[1]
    return str(metadata.get("filesystem_operation") or "").strip()


def _resume_hints_for(*, operation: str, missing_args: tuple[str, ...], user_message: str) -> tuple[str, ...]:
    if operation == "copy_file" and "dst" in missing_args:
        return ("补充具体目标文件名或提交结构化 operation_intent 后继续",)
    if operation in {"move_file", "rename_file"} and "dst" in missing_args:
        return ("补充新的文件名或目标路径",)
    return ("补充缺失参数后继续执行原工具任务",)


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item or "").strip())


__all__ = [
    "PENDING_OPERATION_METADATA_KEY",
    "PendingOperationFrame",
    "build_pending_operation_from_tool_input_completion",
    "clear_pending_operation",
    "pending_operation_from_metadata",
    "with_pending_operation",
]
