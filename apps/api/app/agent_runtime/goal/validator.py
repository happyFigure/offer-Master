from __future__ import annotations

from pathlib import PureWindowsPath
from typing import Any

from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY
from app.agent_runtime.goal.schemas import GoalState, GoalValidationResult
from app.agent_runtime.skills.filesystem_operation_catalog import FilesystemOperationSpec, get_filesystem_operation_spec


def validate_goal_completion(
    *,
    goal_state: GoalState | None,
    tool_name: str,
    tool_input: dict[str, Any] | None,
    result_payload: dict[str, Any] | None,
    semantic_profile: Any | None = None,
) -> GoalValidationResult:
    """Judge whether a tool result completed the user's goal, not just whether the tool ran."""

    if goal_state is None or not goal_state.expected_operation:
        return GoalValidationResult(
            completed=_payload_ok(result_payload),
            advanced=_payload_ok(result_payload),
            recoverable=False,
            reason="当前轮次没有结构化目标，沿用工具执行状态。",
            next_action="final_answer" if _payload_ok(result_payload) else "select_alternative_tool",
        )
    if goal_state.intent == "filesystem_operation" or _looks_like_filesystem_goal(goal_state=goal_state, tool_name=tool_name, result_payload=result_payload or {}):
        return _validate_filesystem_goal(
            goal_state=goal_state,
            tool_name=tool_name,
            tool_input=tool_input or {},
            result_payload=result_payload or {},
        )
    if semantic_profile is not None:
        return _validate_semantic_profile_goal(
            goal_state=goal_state,
            tool_name=tool_name,
            result_payload=result_payload or {},
            semantic_profile=semantic_profile,
        )
    return GoalValidationResult(
        completed=_payload_ok(result_payload),
        advanced=_payload_ok(result_payload),
        recoverable=False,
        reason="当前目标类型暂未配置专用校验器，沿用工具执行状态。",
        next_action="final_answer" if _payload_ok(result_payload) else "select_alternative_tool",
    )


def _looks_like_filesystem_goal(*, goal_state: GoalState, tool_name: str, result_payload: dict[str, Any]) -> bool:
    expected = str(goal_state.expected_operation or "").strip()
    if get_filesystem_operation_spec(expected) is None:
        return False
    result_tool = str(result_payload.get("tool_name") or "").strip()
    result_operation = str(result_payload.get("operation") or "").strip()
    # A short follow-up such as "把名字改成..." can start as normal_chat, then
    # become a file operation through ActiveFileContext. The expected operation
    # is the stronger signal, so filesystem goals must still receive postcheck
    # validation even if the original top-level intent label was not rewritten.
    return tool_name == FILESYSTEM_SKILL_CAPABILITY or result_tool == FILESYSTEM_SKILL_CAPABILITY or get_filesystem_operation_spec(result_operation) is not None


def _validate_semantic_profile_goal(
    *,
    goal_state: GoalState,
    tool_name: str,
    result_payload: dict[str, Any],
    semantic_profile: Any,
) -> GoalValidationResult:
    ok = _payload_ok(result_payload)
    profile_intent = str(getattr(semantic_profile, "intent", "") or "").strip()
    profile_target_type = str(getattr(semantic_profile, "target_type", "") or "").strip()
    expected = str(goal_state.expected_operation or "").strip()
    expected_target_type = str(goal_state.target_type or "").strip()

    # For ordinary tools the result may not contain an explicit operation field,
    # so the registered semantic profile acts as the deterministic operation tag.
    actual_operation = _first_text(
        _result_operation(result_payload),
        profile_intent,
        tool_name,
    )
    if expected and actual_operation != expected:
        return GoalValidationResult(
            completed=False,
            advanced=False,
            recoverable=True,
            reason=f"用户目标需要 {expected}，但本次工具语义是 {actual_operation or 'unknown'}。",
            next_action="continue_loop",
            suggested_operation=expected,
        )
    if expected_target_type and profile_target_type and expected_target_type != profile_target_type:
        return GoalValidationResult(
            completed=False,
            advanced=False,
            recoverable=True,
            reason=f"用户目标对象是 {expected_target_type}，但工具作用对象是 {profile_target_type}。",
            next_action="continue_loop",
            suggested_operation=expected or None,
        )

    missing_fields = _missing_result_contract_fields(result_payload, semantic_profile)
    if ok and not missing_fields:
        return GoalValidationResult(
            completed=True,
            advanced=True,
            recoverable=False,
            reason="工具执行成功，且返回结果满足工具语义档案的成功标准。",
            next_action="final_answer",
            suggested_operation=expected or profile_intent or None,
        )
    if ok and missing_fields:
        return GoalValidationResult(
            completed=False,
            advanced=True,
            recoverable=True,
            reason="工具执行成功，但返回结果缺少语义档案要求的关键字段，不能直接认定用户目标完成。",
            next_action="continue_loop",
            suggested_operation=expected or profile_intent or None,
            missing_information=tuple(missing_fields),
        )
    return GoalValidationResult(
        completed=False,
        advanced=False,
        recoverable=True,
        reason="工具执行失败，需根据语义档案的失败恢复提示继续处理。",
        next_action="select_alternative_tool",
        suggested_operation=expected or profile_intent or None,
    )


def _validate_filesystem_goal(
    *,
    goal_state: GoalState,
    tool_name: str,
    tool_input: dict[str, Any],
    result_payload: dict[str, Any],
) -> GoalValidationResult:
    expected = str(goal_state.expected_operation or "")
    actual = _result_operation(result_payload)
    spec = get_filesystem_operation_spec(expected)
    if spec is None:
        return GoalValidationResult(
            completed=False,
            advanced=False,
            recoverable=True,
            reason=f"文件系统目标暂不支持校验操作：{expected}。",
            next_action="ask_user",
            suggested_capability=_filesystem_capability(tool_name),
            missing_information=("filesystem_operation",),
        )
    if spec.goal_kind == "file_to_file":
        return _validate_file_to_file_goal(spec=spec, goal_state=goal_state, actual_operation=actual, result_payload=result_payload)
    if spec.goal_kind == "replace_text":
        return _validate_replace_text_goal(goal_state=goal_state, actual_operation=actual, result_payload=result_payload)
    if spec.goal_kind in {"path_exists", "read_content"}:
        return _validate_operation_only_goal(spec=spec, actual_operation=actual, result_payload=result_payload)
    return GoalValidationResult(
        completed=False,
        advanced=False,
        recoverable=True,
        reason=f"文件系统目标暂不支持校验类型：{spec.goal_kind}。",
        next_action="ask_user",
        suggested_capability=_filesystem_capability(tool_name),
        missing_information=("filesystem_operation",),
    )


def _validate_file_to_file_goal(
    *,
    spec: FilesystemOperationSpec,
    goal_state: GoalState,
    actual_operation: str,
    result_payload: dict[str, Any],
) -> GoalValidationResult:
    ok = _payload_ok(result_payload)
    expected_target = str(goal_state.target.get(spec.target_match_key) or "").strip()
    if not expected_target:
        return GoalValidationResult(
            completed=False,
            advanced=False,
            recoverable=True,
            reason=f"{spec.summary}缺少目标信息。",
            next_action="ask_user",
            suggested_capability=FILESYSTEM_SKILL_CAPABILITY,
            suggested_operation=spec.operation,
            missing_information=(spec.target_match_key,),
        )
    if actual_operation != spec.operation:
        return GoalValidationResult(
            completed=False,
            advanced=False,
            recoverable=True,
            reason=f"用户目标是{spec.summary}，但实际执行的是 {actual_operation or 'unknown'}。",
            next_action="continue_loop",
            suggested_capability=FILESYSTEM_SKILL_CAPABILITY,
            suggested_operation=spec.operation,
        )
    postcheck = _filesystem_postcheck(result_payload)
    if ok and postcheck is not None and postcheck.get("completed") is False:
        # ok=true only proves the script returned successfully. Filesystem
        # mutations also need a postcondition proof, otherwise a fake/partial
        # script result could be mistaken for the user's completed task.
        return GoalValidationResult(
            completed=False,
            advanced=False,
            recoverable=True,
            reason=f"{spec.summary}脚本返回成功，但文件状态复核未通过：{postcheck.get('reason') or 'postcheck_failed'}。",
            next_action="continue_loop",
            suggested_capability=FILESYSTEM_SKILL_CAPABILITY,
            suggested_operation=spec.operation,
        )
    actual_target = _first_text(
        _nested(result_payload, "arguments", spec.result_path_arg),
        result_payload.get(spec.result_path_arg),
        _nested(result_payload, "result", spec.result_path_arg),
        _nested(result_payload, "result", "result", spec.result_path_arg),
    )
    if ok and _target_matches(actual_target, expected_target, mode=spec.target_match_mode):
        return GoalValidationResult(
            completed=True,
            advanced=True,
            recoverable=False,
            reason=f"{spec.summary}结果与用户目标一致。",
            next_action="final_answer",
            suggested_capability=FILESYSTEM_SKILL_CAPABILITY,
            suggested_operation=spec.operation,
        )
    return GoalValidationResult(
        completed=False,
        advanced=ok,
        recoverable=True,
        reason=f"{spec.summary}工具已返回结果，但目标没有确认匹配。",
        next_action="continue_loop" if ok else "select_alternative_tool",
        suggested_capability=FILESYSTEM_SKILL_CAPABILITY,
        suggested_operation=spec.operation,
    )


def _validate_replace_text_goal(*, goal_state: GoalState, actual_operation: str, result_payload: dict[str, Any]) -> GoalValidationResult:
    ok = _payload_ok(result_payload)
    if actual_operation != "replace_text":
        return GoalValidationResult(
            completed=False,
            advanced=False,
            recoverable=True,
            reason=f"用户目标是替换文件内容，但实际执行的是 {actual_operation or 'unknown'}。",
            next_action="continue_loop",
            suggested_capability=FILESYSTEM_SKILL_CAPABILITY,
            suggested_operation="replace_text",
        )
    replacement_count = _replacement_count(result_payload)
    if ok and _payload_no_op(result_payload):
        return GoalValidationResult(
            completed=True,
            advanced=True,
            recoverable=False,
            reason="替换前后的文本完全相同，文件已经满足当前目标，无需再次执行写操作。",
            next_action="final_answer",
            suggested_capability=FILESYSTEM_SKILL_CAPABILITY,
            suggested_operation="replace_text",
        )
    if ok and replacement_count > 0:
        return GoalValidationResult(
            completed=True,
            advanced=True,
            recoverable=False,
            reason="文件内容替换次数大于 0，目标已推进并完成。",
            next_action="final_answer",
            suggested_capability=FILESYSTEM_SKILL_CAPABILITY,
            suggested_operation="replace_text",
        )
    if ok and replacement_count == 0:
        return GoalValidationResult(
            completed=False,
            advanced=False,
            recoverable=True,
            reason="替换工具执行成功，但替换次数为 0，说明目标文本可能没有找到。",
            next_action="ask_user_or_search_text",
            suggested_capability=FILESYSTEM_SKILL_CAPABILITY,
            suggested_operation="replace_text",
            missing_information=("old_text",),
        )
    return GoalValidationResult(
        completed=False,
        advanced=False,
        recoverable=True,
        reason="替换文件内容工具执行失败。",
        next_action="select_alternative_tool",
        suggested_capability=FILESYSTEM_SKILL_CAPABILITY,
        suggested_operation="replace_text",
    )


def _validate_operation_only_goal(*, spec: FilesystemOperationSpec, actual_operation: str, result_payload: dict[str, Any]) -> GoalValidationResult:
    ok = _payload_ok(result_payload)
    completed = ok and actual_operation == spec.operation
    return GoalValidationResult(
        completed=completed,
        advanced=completed,
        recoverable=not completed,
        reason=f"{spec.summary}目标已完成。" if completed else f"用户目标是{spec.summary}，但实际工具结果不匹配。",
        next_action="final_answer" if completed else "continue_loop",
        suggested_capability=FILESYSTEM_SKILL_CAPABILITY,
        suggested_operation=spec.operation,
    )


def _payload_ok(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    if "ok" in payload:
        return bool(payload.get("ok"))
    envelope = payload.get("result_envelope") if isinstance(payload.get("result_envelope"), dict) else {}
    status = str(envelope.get("status") or payload.get("status") or "").lower()
    return status == "succeeded" if status else True


def _result_operation(payload: dict[str, Any]) -> str:
    return _first_text(payload.get("operation"), _nested(payload, "result", "operation"), _nested(payload, "result", "result", "operation"))


def _replacement_count(payload: dict[str, Any]) -> int:
    value = _first_value(
        payload.get("replacement_count"),
        payload.get("replacements"),
        _nested(payload, "result", "replacement_count"),
        _nested(payload, "result", "replacements"),
        _nested(payload, "result", "result", "replacement_count"),
    )
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _payload_no_op(payload: dict[str, Any]) -> bool:
    # Some filesystem actions complete without mutating the file. Keep this as
    # an explicit contract instead of guessing from replacement_count == 0,
    # because zero replacements can also mean "target text not found".
    return bool(
        _first_value(
            payload.get("no_op"),
            _nested(payload, "result", "no_op"),
            _nested(payload, "result", "result", "no_op"),
        )
    )


def _filesystem_postcheck(payload: dict[str, Any]) -> dict[str, Any] | None:
    trace = payload.get("filesystem_trace") if isinstance(payload.get("filesystem_trace"), dict) else None
    if trace is None:
        trace = _nested(payload, "result", "filesystem_trace")
    if not isinstance(trace, dict):
        return None
    postcheck = trace.get("postcheck")
    return postcheck if isinstance(postcheck, dict) else None


def _missing_result_contract_fields(payload: dict[str, Any], semantic_profile: Any) -> list[str]:
    contract = getattr(semantic_profile, "result_contract", None)
    if not isinstance(contract, dict):
        return []
    required_fields = contract.get("required_fields")
    if not isinstance(required_fields, list):
        return []
    missing: list[str] = []
    for field_path in required_fields:
        path = str(field_path or "").strip()
        if not path:
            continue
        if _field_path_value(payload, path) is None:
            missing.append(path)
    return missing


def _field_path_value(payload: dict[str, Any], field_path: str) -> Any:
    current: Any = payload
    for key in field_path.split("."):
        if not key:
            return None
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _filename(value: str) -> str:
    if not value:
        return ""
    return PureWindowsPath(value).name or value.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def _target_matches(actual: str, expected: str, *, mode: str) -> bool:
    if mode == "filename":
        return _filename(actual) == expected
    if mode == "path":
        return _same_path_text(actual, expected)
    return actual == expected


def _same_path_text(left: str, right: str) -> bool:
    return left.replace("\\", "/").rstrip("/").lower() == right.replace("\\", "/").rstrip("/").lower()


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_text(*values: Any) -> str:
    value = _first_value(*values)
    return str(value or "").strip()


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _filesystem_capability(tool_name: str) -> str:
    return tool_name if tool_name == FILESYSTEM_SKILL_CAPABILITY else FILESYSTEM_SKILL_CAPABILITY


__all__ = ["validate_goal_completion"]
