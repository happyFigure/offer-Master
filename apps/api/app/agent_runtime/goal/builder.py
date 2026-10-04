from __future__ import annotations

from typing import Any

from app.agent_runtime.context.file_operation_policy import (
    generate_rename_destination_from_operation_intent,
    resolve_copy_destination_from_operation_intent,
)
from app.agent_runtime.goal.schemas import GoalState
from app.agent_runtime.skills.filesystem_operation_catalog import get_filesystem_operation_spec


_SEMANTIC_GOAL_FALLBACKS: dict[str, dict[str, Any]] = {
    "local_company_database_list": {
        "intent": "list_local_companies",
        "target_type": "company_list",
        "success_criteria": ("tool_result_ok", "result_field_present:companies"),
    }
}


def build_goal_state(*, user_message: str, intent: str, context_metadata: dict[str, Any] | None = None) -> GoalState:
    """Build a goal from structured intent; never infer file operations from prose."""

    context = context_metadata or {}
    if intent == "filesystem_operation" or context.get("filesystem_operation"):
        return _filesystem_goal_state(intent="filesystem_operation", context={**context, "user_message": user_message})

    semantic_goal = _semantic_goal_state(intent=intent, context=context)
    if semantic_goal is not None:
        return semantic_goal
    return GoalState(intent=intent, source="runtime_goal_builder_v1")


def _filesystem_goal_state(*, intent: str, context: dict[str, Any]) -> GoalState:
    operation = str(context.get("filesystem_operation") or "").strip()
    spec = get_filesystem_operation_spec(operation)
    if spec is None:
        return GoalState(intent=intent, target_type="file", source="runtime_structured_filesystem_intent_v2")

    active_file = context.get("active_file") if isinstance(context.get("active_file"), dict) else {}
    source_path = str(active_file.get("path") or "").strip()
    operation_intent = context.get("operation_intent") if isinstance(context.get("operation_intent"), dict) else {}

    if operation in {"read_file", "path_exists"}:
        target = {"path": source_path}
    elif operation in {"copy_file", "rename_file"}:
        target_path = (
            resolve_copy_destination_from_operation_intent(operation_intent, source_path)
            if operation == "copy_file"
            else generate_rename_destination_from_operation_intent(operation_intent, source_path)
        )
        target = {"source_path": source_path, "target_path": target_path}
        if operation == "rename_file":
            target["target_name"] = _filename(target_path)
    elif operation == "replace_text":
        target = {"path": source_path}
    else:
        target = {}

    user_message = str(context.get("user_message") or "")
    answer_intent = None
    answer_policy: dict[str, Any] = {}
    if operation == "read_file" and any(marker in user_message for marker in ("总结", "概括", "摘要", "归纳", "主要内容")):
        answer_intent = "summarize_document"
        answer_policy = {"mode": "summary", "language": "zh-CN", "do_not_echo_full_content": True}

    return GoalState(
        intent=intent,
        target_type=spec.target_type,
        expected_operation=operation,
        target=target,
        success_criteria=spec.success_criteria,
        answer_intent=answer_intent,
        answer_policy=answer_policy,
        source="runtime_structured_filesystem_intent_v2",
    )


def _semantic_goal_state(*, intent: str, context: dict[str, Any]) -> GoalState | None:
    profile = _semantic_profile_from_context(intent=intent, context=context) or _SEMANTIC_GOAL_FALLBACKS.get(intent)
    if profile is None:
        return None
    operation = _profile_text(profile, "intent")
    if not operation:
        return None
    return GoalState(
        intent=intent,
        target_type=_profile_text(profile, "target_type") or "tool_result",
        expected_operation=operation,
        success_criteria=_profile_success_criteria(profile),
        source="runtime_semantic_profile_v1",
    )


def _semantic_profile_from_context(*, intent: str, context: dict[str, Any]) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for metadata in _capability_metadata_entries(context):
        if not isinstance(metadata, dict):
            continue
        allowed_intents = {str(item) for item in metadata.get("allowed_intents") or [] if str(item).strip()}
        if allowed_intents and intent not in allowed_intents:
            continue
        profile = metadata.get("semantic_profile")
        if isinstance(profile, dict) and _profile_text(profile, "intent"):
            matches.append(profile)
    return matches[0] if len(matches) == 1 else None


def _capability_metadata_entries(context: dict[str, Any]) -> list[Any]:
    direct = context.get("capability_metadata")
    if isinstance(direct, list):
        return direct
    context_pack = context.get("context_pack")
    if isinstance(context_pack, dict) and isinstance(context_pack.get("capability_metadata"), list):
        return list(context_pack["capability_metadata"])
    return []


def _profile_text(profile: dict[str, Any], key: str) -> str:
    return str(profile.get(key) or "").strip()


def _profile_success_criteria(profile: dict[str, Any]) -> tuple[str, ...]:
    criteria = profile.get("success_criteria")
    if isinstance(criteria, str):
        return (criteria,) if criteria.strip() else ("tool_result_ok",)
    if isinstance(criteria, (list, tuple, set)):
        cleaned = tuple(str(item).strip() for item in criteria if str(item).strip())
        return cleaned or ("tool_result_ok",)
    return ("tool_result_ok",)


def _filename(path: str) -> str:
    normalized = str(path or "").strip()
    index = max(normalized.rfind("/"), normalized.rfind("\\"))
    return normalized[index + 1 :] if index >= 0 else normalized


__all__ = ["build_goal_state"]
