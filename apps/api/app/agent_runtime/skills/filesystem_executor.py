from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from app.agent_runtime.agent_as_tool import (
    FILESYSTEM_SKILL_CAPABILITY,
    FILESYSTEM_SKILL_EXECUTOR_ID,
    AgentCapabilityDefinition,
    AgentRuntimeContext,
    AgentTask,
    StandardAgentResult,
)
from app.agent_runtime.context.file_context import (
    complete_copy_file_arguments,
    complete_move_file_arguments,
    extract_local_file_references,
)
from app.agent_runtime.skills.filesystem_operation_catalog import (
    FilesystemOperationSpec,
    build_filesystem_skill_input_schema,
    build_operation_approval_payload,
    format_operation_approval_summary,
    format_operation_execution_summary,
    get_filesystem_operation_spec,
)
from app.agent_runtime.skills.filesystem_no_dead_end import build_filesystem_no_dead_end_outcome
from app.agent_runtime.tool_registry import (
    FILESYSTEM_REPLACE_TEXT_TOOL,
    AgentToolCandidateProfile,
    run_filesystem_skill_script,
)
from app.agent_runtime.skills.filesystem_semantic_admission import validate_filesystem_semantic_admission
from app.agent_runtime.tool_input_completion import complete_tool_input


READ_FILE_OPERATION = "read_file"
PATH_EXISTS_OPERATION = "path_exists"
RENAME_FILE_OPERATION = "rename_file"
REPLACE_TEXT_OPERATION = "replace_text"
COPY_FILE_OPERATION = "copy_file"
FILESYSTEM_POSTCHECK_ERROR_CODE = "FILESYSTEM_POSTCHECK_FAILED"
SEMANTIC_ADMISSION_ERROR_CODE = "SEMANTIC_ADMISSION_REJECTED"

logger = logging.getLogger(__name__)

_FILE_TO_FILE_COMPLETERS = {
    COPY_FILE_OPERATION: complete_copy_file_arguments,
    RENAME_FILE_OPERATION: complete_move_file_arguments,
}

REPLACE_TEXT_INPUT_SCHEMA = {
    "type": "object",
    "required": ["path", "old_text", "new_text"],
    "properties": {
        "path": {"type": "string"},
        "old_text": {"type": "string"},
        "new_text": {"type": "string"},
        "encoding": {"type": "string"},
        "count": {"type": "integer"},
    },
    "additionalProperties": False,
}


@dataclass(frozen=True)
class FilesystemSkillExecutor:
    """Executes the high-level filesystem Skill without exposing inner scripts to the main agent."""

    script_root: str | Path | None = None
    session_provider: Any | None = None
    executor_id: str = FILESYSTEM_SKILL_EXECUTOR_ID

    def capabilities(self) -> list[AgentCapabilityDefinition]:
        return [_filesystem_skill_capability_definition()]

    def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
        user_task = _user_task(task)
        context_metadata = _context_metadata(task, context)
        # The model must submit the inner operation as structured data. The
        # runtime deliberately does not classify the user's prose here.
        explicit_operation = _requested_operation(task.input_payload) or ""
        operation = explicit_operation
        intent_resolution = {
            "operation": operation,
            "source": "model_structured",
            "operation_intent": dict(task.input_payload.get("operation_intent") or {}),
        }
        logger.info(
            "Filesystem skill structured intent received: operation=%s active_file=%s input_keys=%s",
            operation,
            _active_file_path(context_metadata),
            sorted(task.input_payload.keys()),
        )
        if not operation:
            return StandardAgentResult(
                status="failed",
                summary="filesystem Skill 缺少结构化 operation，runtime 不会从用户文本猜测文件动作。",
                missing_information=["operation"],
                # The user has already delegated the file task. The missing
                # inner operation is model-owned semantic work, so keep the
                # model loop alive instead of turning this into a clarification.
                requires_user_action=False,
                raw_result={
                    "tool_name": FILESYSTEM_SKILL_CAPABILITY,
                    "ok": False,
                    "error_code": "STRUCTURED_OPERATION_REQUIRED",
                    "missing_information": ["operation"],
                    "recoverable": True,
                    "next_action": "continue_model_loop",
                    "intent_resolution": intent_resolution,
                },
            )
        if operation == "unknown":
            outcome = build_filesystem_no_dead_end_outcome(
                operation=operation,
                reason="模型提交了未知的 filesystem operation",
                candidates=[],
                next_action="ask_user",
            )
            return StandardAgentResult(
                status="failed",
                summary=outcome["summary"],
                missing_information=["filesystem_operation"],
                raw_result={"tool_name": FILESYSTEM_SKILL_CAPABILITY, **outcome, "intent_resolution": intent_resolution},
            )
        spec = get_filesystem_operation_spec(operation)
        if spec is None:
            outcome = build_filesystem_no_dead_end_outcome(
                operation=operation,
                reason="无法判断应该执行哪个文件动作",
                candidates=[],
                next_action="ask_user",
            )
            return StandardAgentResult(
                status="failed",
                summary=outcome["summary"],
                missing_information=["filesystem_operation"],
                raw_result={"tool_name": FILESYSTEM_SKILL_CAPABILITY, **outcome, "intent_resolution": intent_resolution},
            )
        if spec.goal_kind == "file_to_file":
            return self._file_to_file_operation(
                spec=spec,
                task=task,
                user_task=user_task,
                context_metadata=context_metadata,
                context=context,
                intent_resolution=intent_resolution,
            )
        if spec.goal_kind == "replace_text":
            return self._replace_text(task=task, user_task=user_task, context_metadata=context_metadata, context=context, intent_resolution=intent_resolution)
        if spec.goal_kind == "path_exists":
            return self._path_exists(task=task, user_task=user_task, context_metadata=context_metadata, context=context, intent_resolution=intent_resolution)
        if spec.goal_kind == "read_content":
            return self._read_file(task=task, user_task=user_task, context_metadata=context_metadata, context=context, intent_resolution=intent_resolution)
        outcome = build_filesystem_no_dead_end_outcome(
            operation=operation,
            reason="文件动作 catalog 存在，但 executor 暂不支持该 goal_kind",
            candidates=[],
            next_action="ask_user",
        )
        return StandardAgentResult(
            status="failed",
            summary=outcome["summary"],
            missing_information=["filesystem_operation"],
            raw_result={"tool_name": FILESYSTEM_SKILL_CAPABILITY, **outcome, "intent_resolution": intent_resolution},
        )

    def _file_to_file_operation(
        self,
        *,
        spec: FilesystemOperationSpec,
        task: AgentTask,
        user_task: str,
        context_metadata: dict[str, Any],
        context: AgentRuntimeContext,
        intent_resolution: dict[str, Any] | None = None,
    ) -> StandardAgentResult:
        seed_input = {key: value for key, value in task.input_payload.items() if key in set(spec.input_fields)}
        completer = _FILE_TO_FILE_COMPLETERS.get(spec.operation)
        arguments = completer(tool_input=seed_input, user_message=user_task, context=context_metadata) if completer else dict(seed_input)
        missing = [field for field in spec.required_args if not str(arguments.get(field) or "").strip()]
        logger.info(
            "Filesystem skill arguments completed: operation=%s arguments=%s missing=%s",
            spec.operation,
            arguments,
            missing,
        )
        content_based_name_requested = _is_content_based_name_policy(arguments.get("operation_intent"))
        content_read_available = _has_completed_content_read(
            context_metadata,
            source_path=str(arguments.get("src") or arguments.get("path") or "").strip(),
        )
        if missing:
            reason = f"{spec.summary}缺少必要信息：{', '.join(missing)}"
            requires_content_read = _requires_content_read_for_name(
                operation=spec.operation,
                tool_input=arguments,
                context_metadata=context_metadata,
            )
            if content_based_name_requested and content_read_available and not _has_model_name_intent(arguments.get("operation_intent")):
                return _content_based_name_required_result(
                    spec=spec,
                    arguments=arguments,
                    intent_resolution=intent_resolution,
                )
            # A missing destination is model-owned semantic work. Keep the
            # native tool loop alive so the model can supply dst/name_intent;
            # do not turn a delegated naming request into a user-facing dead end.
            model_owned_missing = any(field in {"dst", "operation_intent", "name_intent"} for field in missing)
            next_action = "read_before_write" if requires_content_read else ("continue_model_loop" if model_owned_missing else "wait_user_input")
            # Missing arguments are not script failures. They are resumable
            # states that the outer runtime can persist and complete later.
            outcome = build_filesystem_no_dead_end_outcome(
                operation=spec.operation,
                reason=(
                    f"{spec.summary}需要先读取文件内容，让模型根据真实内容选择目标文件名"
                    if requires_content_read
                    else reason
                ),
                next_action=next_action,
                missing_information=tuple(missing),
            )
            return StandardAgentResult(
                status="failed",
                summary=outcome["summary"],
                missing_information=missing,
                requires_user_action=not requires_content_read and not model_owned_missing,
                raw_result={
                    "tool_name": FILESYSTEM_SKILL_CAPABILITY,
                    **outcome,
                    "error": reason,
                    "error_code": "CONTENT_BASED_NAME_REQUIRES_READ" if requires_content_read else "MISSING_REQUIRED_ARGUMENT",
                    "missing_args": list(missing),
                    "retryable": bool(requires_content_read or model_owned_missing),
                    "arguments": dict(arguments),
                    "read_before_write": {
                        "required": requires_content_read,
                        "operation": READ_FILE_OPERATION,
                        "path": str(arguments.get("src") or arguments.get("path") or ""),
                    }
                    if requires_content_read
                    else None,
                    "intent_resolution": intent_resolution,
                },
            )

        if content_based_name_requested and not content_read_available:
            outcome = build_filesystem_no_dead_end_outcome(
                operation=spec.operation,
                reason=f"{spec.summary}需要先读取源文件内容，再由模型根据真实内容提交文件名",
                next_action="read_before_write",
                missing_information=("name_intent",),
            )
            return StandardAgentResult(
                status="failed",
                summary=outcome["summary"],
                missing_information=["name_intent"],
                requires_user_action=False,
                raw_result={
                    "tool_name": FILESYSTEM_SKILL_CAPABILITY,
                    **outcome,
                    "error_code": "CONTENT_BASED_NAME_REQUIRES_READ",
                    "retryable": True,
                    "arguments": dict(arguments),
                    "read_before_write": {
                        "required": True,
                        "operation": READ_FILE_OPERATION,
                        "path": str(arguments.get("src") or arguments.get("path") or ""),
                    },
                    "intent_resolution": intent_resolution,
                },
            )

        if content_based_name_requested and not _has_model_name_intent(arguments.get("operation_intent")):
            return _content_based_name_required_result(
                spec=spec,
                arguments=arguments,
                intent_resolution=intent_resolution,
            )

        # This checkpoint runs after mechanical slot filling but before the
        # approval card or script execution. It keeps filler phrases such as
        # "改一下" from becoming destructive filesystem targets like "一下.tex".
        semantic_admission = validate_filesystem_semantic_admission(
            operation=spec.operation,
            arguments=arguments,
            user_message=user_task,
        )
        requires_fresh_confirmation = False
        if not semantic_admission.allowed:
            logger.info(
                "Filesystem semantic admission rejected operation=%s reason_code=%s src=%s dst=%s",
                spec.operation,
                semantic_admission.reason_code,
                _source_path_for_operation(spec, arguments),
                _target_path_for_operation(spec, arguments),
            )
            outcome = build_filesystem_no_dead_end_outcome(
                operation=spec.operation,
                reason=semantic_admission.reason,
                next_action=semantic_admission.next_action,
                missing_information=("dst",),
            )
            return StandardAgentResult(
                status="failed",
                summary=outcome["summary"],
                observation=outcome["summary"],
                missing_information=["dst"],
                raw_result={
                    "tool_name": FILESYSTEM_SKILL_CAPABILITY,
                    **outcome,
                    "error": semantic_admission.reason,
                    "error_code": SEMANTIC_ADMISSION_ERROR_CODE,
                    "retryable": True,
                    "arguments": dict(arguments),
                    "semantic_admission": semantic_admission.to_metadata_dict(),
                    "intent_resolution": intent_resolution,
                },
            )
        if semantic_admission.corrected_arguments:
            logger.info(
                "Filesystem semantic admission repaired operation=%s reason_code=%s src=%s dst=%s corrected_dst=%s",
                spec.operation,
                semantic_admission.reason_code,
                _source_path_for_operation(spec, arguments),
                _target_path_for_operation(spec, arguments),
                _target_path_for_operation(spec, semantic_admission.corrected_arguments),
            )
            arguments = dict(semantic_admission.corrected_arguments)
            requires_fresh_confirmation = True

        approval_payload = _approval_payload(spec, arguments)
        if spec.requires_confirmation and (requires_fresh_confirmation or not bool(context.permission_scope.get("user_confirmed"))):
            return StandardAgentResult(
                status="waiting_user",
                summary=f"filesystem Skill 需要用户确认后才能执行：{spec.summary}",
                requires_user_action=True,
                raw_result={
                    "tool_name": FILESYSTEM_SKILL_CAPABILITY,
                    "ok": False,
                    "operation": spec.operation,
                    "intent_resolution": intent_resolution,
                    "semantic_admission": semantic_admission.to_metadata_dict(),
                    "approval_payload": approval_payload,
                    "approval_request": {
                        "approval_type": "skill_operation",
                        "tool_name": FILESYSTEM_SKILL_CAPABILITY,
                        "tool_input": dict(arguments),
                        "operation": spec.operation,
                        "risk_level": spec.risk_level,
                        "reason": spec.approval_reason,
                        "suggested_user_message": _approval_summary(spec, approval_payload),
                    },
                },
            )

        return self._run_internal_action(spec, arguments, intent_resolution=intent_resolution)

    def _replace_text(
        self,
        *,
        task: AgentTask,
        user_task: str,
        context_metadata: dict[str, Any],
        context: AgentRuntimeContext,
        intent_resolution: dict[str, Any] | None = None,
    ) -> StandardAgentResult:
        seed_input = {
            key: value
            for key, value in task.input_payload.items()
            if key in {"path", "old_text", "new_text", "encoding", "count"}
        }
        if not str(seed_input.get("path") or "").strip():
            seed_input["path"] = _active_file_path(context_metadata)
        completion = complete_tool_input(
            tool_name=FILESYSTEM_REPLACE_TEXT_TOOL,
            tool_input=seed_input,
            input_schema=REPLACE_TEXT_INPUT_SCHEMA,
            user_message=user_task,
            context=context_metadata,
        )
        arguments = dict(completion.tool_input)
        if "encoding" not in arguments:
            arguments["encoding"] = "utf-8"
        if "count" not in arguments:
            arguments["count"] = 0
        missing = [field for field in ("path", "old_text", "new_text") if not str(arguments.get(field) or "").strip()]
        if missing:
            return StandardAgentResult(
                status="failed",
                summary=f"替换文件内容缺少必要信息：{', '.join(missing)}。",
                missing_information=missing,
                raw_result={"tool_name": FILESYSTEM_SKILL_CAPABILITY, "ok": False, "operation": REPLACE_TEXT_OPERATION, "intent_resolution": intent_resolution},
            )

        old_text = str(arguments.get("old_text") or "")
        new_text = str(arguments.get("new_text") or "")
        if old_text == new_text:
            # Replacing text with the same value is not a high-risk write: there
            # is nothing to mutate, so creating an approval card would only make
            # the workflow look stuck after the user confirms it.
            summary = f"没有执行替换：新旧内容相同（{old_text}），文件已经满足当前目标。"
            return StandardAgentResult(
                status="succeeded",
                summary=summary,
                requires_user_action=False,
                raw_result={
                    "tool_name": FILESYSTEM_SKILL_CAPABILITY,
                    "ok": True,
                    "operation": REPLACE_TEXT_OPERATION,
                    "intent_resolution": intent_resolution,
                    "no_op": True,
                    "reason": "old_text_equals_new_text",
                    "summary": summary,
                    "arguments": dict(arguments),
                    "replacement_count": 0,
                    "result": {
                        "tool_name": FILESYSTEM_REPLACE_TEXT_TOOL,
                        "ok": True,
                        "result": {
                            "operation": REPLACE_TEXT_OPERATION,
                            "path": str(arguments.get("path") or ""),
                            "replacement_count": 0,
                            "no_op": True,
                        },
                    },
                },
            )

        spec = get_filesystem_operation_spec(REPLACE_TEXT_OPERATION)
        approval_payload = _approval_payload(spec, arguments)
        if not bool(context.permission_scope.get("user_confirmed")):
            return StandardAgentResult(
                status="waiting_user",
                summary=f"filesystem Skill 需要用户确认后才能执行：{spec.summary if spec is not None else '替换文件内容'}",
                requires_user_action=True,
                raw_result={
                    "tool_name": FILESYSTEM_SKILL_CAPABILITY,
                    "ok": False,
                    "operation": REPLACE_TEXT_OPERATION,
                    "intent_resolution": intent_resolution,
                    "approval_payload": approval_payload,
                    "approval_request": {
                        "approval_type": "skill_operation",
                        "tool_name": FILESYSTEM_SKILL_CAPABILITY,
                        "tool_input": dict(arguments),
                        "operation": REPLACE_TEXT_OPERATION,
                        "risk_level": spec.risk_level if spec is not None else "high",
                        "reason": spec.approval_reason if spec is not None else "替换本地文件内容属于高风险写操作，需要用户确认。",
                        "suggested_user_message": _approval_summary(spec, approval_payload),
                    },
                },
            )

        return self._run_internal_action(spec, arguments, intent_resolution=intent_resolution)

    def _path_exists(
        self,
        *,
        task: AgentTask,
        user_task: str,
        context_metadata: dict[str, Any],
        context: AgentRuntimeContext,
        intent_resolution: dict[str, Any] | None = None,
    ) -> StandardAgentResult:
        # Existence checks are read-only, so they should not enter the same
        # approval path as rename/replace write operations.
        path = _path_from_task_context_or_text(task=task, user_task=user_task, context_metadata=context_metadata)
        if not path:
            return StandardAgentResult(
                status="failed",
                summary="检查路径是否存在缺少 path。",
                missing_information=["path"],
                raw_result={"tool_name": FILESYSTEM_SKILL_CAPABILITY, "ok": False, "operation": PATH_EXISTS_OPERATION, "intent_resolution": intent_resolution},
            )
        return self._run_internal_action(get_filesystem_operation_spec(PATH_EXISTS_OPERATION), {"path": path}, intent_resolution=intent_resolution)

    def _read_file(
        self,
        *,
        task: AgentTask,
        user_task: str,
        context_metadata: dict[str, Any],
        context: AgentRuntimeContext,
        intent_resolution: dict[str, Any] | None = None,
    ) -> StandardAgentResult:
        path = str(task.input_payload.get("path") or _active_file_path(context_metadata) or "").strip()
        if not path:
            paths = extract_local_file_references(user_task)
            path = paths[-1] if paths else ""
        if not path:
            return StandardAgentResult(
                status="failed",
                summary="读取文件缺少 path。",
                missing_information=["path"],
                raw_result={"tool_name": FILESYSTEM_SKILL_CAPABILITY, "ok": False, "operation": READ_FILE_OPERATION, "intent_resolution": intent_resolution},
            )
        arguments = {
            "path": path,
            "offset": task.input_payload.get("offset", 0),
            "limit": task.input_payload.get("limit", 200),
            "encoding": task.input_payload.get("encoding", "auto"),
        }
        return self._run_internal_action(get_filesystem_operation_spec(READ_FILE_OPERATION), arguments, intent_resolution=intent_resolution)

    def _run_internal_action(
        self,
        spec: FilesystemOperationSpec | None,
        arguments: dict[str, Any],
        *,
        intent_resolution: dict[str, Any] | None = None,
    ) -> StandardAgentResult:
        if spec is None:
            return StandardAgentResult(
                status="failed",
                summary="filesystem Skill 找不到内部动作定义。",
                raw_result={"tool_name": FILESYSTEM_SKILL_CAPABILITY, "ok": False, "operation": "unknown", "intent_resolution": intent_resolution},
            )
        filesystem_trace = _filesystem_trace_before_execution(spec, arguments)
        raw_payload = run_filesystem_skill_script(
            self._session(),
            tool_name=spec.legacy_tool_name,
            script_name=spec.script_name,
            script_root=self.script_root,
            arguments=arguments,
        )
        ok = bool(raw_payload.get("ok"))
        filesystem_trace["script"] = _filesystem_script_trace(spec, raw_payload)
        postcheck = _filesystem_postcheck(spec, arguments, raw_payload=raw_payload, ok=ok)
        filesystem_trace["postcheck"] = postcheck

        if ok and postcheck.get("completed") is False:
            # Script stdout is only a claim. For write-like filesystem actions,
            # the runtime must verify the observable file state before telling
            # the user that the task is done.
            error = _filesystem_postcheck_error(spec, postcheck)
            return StandardAgentResult(
                status="failed",
                summary=error,
                observation=error,
                raw_result={
                    "tool_name": FILESYSTEM_SKILL_CAPABILITY,
                    "ok": False,
                    "operation": spec.operation,
                    "error": error,
                    "error_code": FILESYSTEM_POSTCHECK_ERROR_CODE,
                    "intent_resolution": intent_resolution,
                    "internal_tool": spec.legacy_tool_name,
                    "internal_script": spec.script_name,
                    "arguments": dict(arguments),
                    "filesystem_trace": filesystem_trace,
                    "result": raw_payload,
                },
            )

        result = raw_payload.get("result") if isinstance(raw_payload.get("result"), dict) else {}
        error = None if ok else str(raw_payload.get("error") or "filesystem Skill 执行失败。")
        return StandardAgentResult(
            status="succeeded" if ok else "failed",
            summary=_execution_summary(spec, raw_payload, ok=ok),
            observation=str(result.get("content") or result.get("stdout") or "")[:2000],
            raw_result={
                "tool_name": FILESYSTEM_SKILL_CAPABILITY,
                "ok": ok,
                "operation": spec.operation,
                "error": error,
                "intent_resolution": intent_resolution,
                "internal_tool": spec.legacy_tool_name,
                "internal_script": spec.script_name,
                "arguments": dict(arguments),
                "filesystem_trace": filesystem_trace,
                "result": raw_payload,
            },
        )

    def _session(self) -> Any:
        if callable(self.session_provider):
            return self.session_provider()
        return None


def classify_filesystem_operation(user_task: str, context_metadata: dict[str, Any]) -> str:
    operation = context_metadata.get("filesystem_operation") or context_metadata.get("operation")
    return str(operation or "unknown").strip() or "unknown"


def _requires_content_read_for_name(
    *,
    operation: str,
    tool_input: dict[str, Any],
    context_metadata: dict[str, Any] | None = None,
) -> bool:
    if operation not in {COPY_FILE_OPERATION, RENAME_FILE_OPERATION}:
        return False
    if not _is_content_based_name_policy(tool_input.get("operation_intent")):
        return False
    return not _has_completed_content_read(
        context_metadata or {},
        source_path=str(tool_input.get("src") or tool_input.get("path") or "").strip(),
    )


def _is_content_based_name_policy(operation_intent: Any) -> bool:
    if not isinstance(operation_intent, dict):
        return False
    return str(operation_intent.get("name_policy") or "").strip().lower() in {
        "content_based",
        "content_based_name",
        "name_from_content",
    }


def _has_model_name_intent(operation_intent: Any) -> bool:
    if not isinstance(operation_intent, dict):
        return False
    name_intent = operation_intent.get("name_intent")
    if not isinstance(name_intent, dict):
        return False
    return any(str(name_intent.get(key) or "").strip() for key in ("filename", "filename_stem", "display_name"))


def _has_completed_content_read(context_metadata: dict[str, Any], *, source_path: str) -> bool:
    last_result = context_metadata.get("last_file_operation_result")
    if not isinstance(last_result, dict):
        return False
    if str(last_result.get("operation") or "").strip() != READ_FILE_OPERATION or last_result.get("completed") is not True:
        return False
    if not source_path:
        return True
    observed_path = str(
        last_result.get("focus_path")
        or last_result.get("source_path")
        or last_result.get("path")
        or ""
    ).strip()
    return bool(observed_path and _same_path_text(observed_path, source_path))


def _content_based_name_required_result(
    *,
    spec: FilesystemOperationSpec,
    arguments: dict[str, Any],
    intent_resolution: dict[str, Any] | None,
) -> StandardAgentResult:
    outcome = build_filesystem_no_dead_end_outcome(
        operation=spec.operation,
        reason="content_based 命名必须由模型提交结构化 name_intent，runtime 不会替模型创作文件名",
        next_action="continue_model_loop",
        missing_information=("name_intent",),
    )
    logger.info(
        "Filesystem content-based name missing model intent: operation=%s src=%s input_keys=%s",
        spec.operation,
        _source_path_for_operation(spec, arguments),
        sorted(arguments.keys()),
    )
    return StandardAgentResult(
        status="failed",
        summary=outcome["summary"],
        missing_information=["name_intent"],
        requires_user_action=False,
        raw_result={
            "tool_name": FILESYSTEM_SKILL_CAPABILITY,
            **outcome,
            "error_code": "CONTENT_BASED_NAME_REQUIRED",
            "retryable": True,
            "arguments": dict(arguments),
            "intent_resolution": intent_resolution,
        },
    )


def _filesystem_skill_capability_definition() -> AgentCapabilityDefinition:
    return AgentCapabilityDefinition(
        capability_id=FILESYSTEM_SKILL_CAPABILITY,
        name="Filesystem Skill",
        description="Use the filesystem Skill for local file operations; the executor chooses internal scripts.",
        executor_id=FILESYSTEM_SKILL_EXECUTOR_ID,
        input_schema=build_filesystem_skill_input_schema(),
        output_schema={"type": "object", "additionalProperties": True},
        kind="skill",
        risk_level="medium",
        supported_intents=("filesystem_operation",),
        allowed_source_types=frozenset({"agent_chat"}),
        candidate_profile=AgentToolCandidateProfile(
            categories=frozenset({"skill_filesystem", "filesystem_operation"}),
            keywords=frozenset({"文件", "文件名", "目录", "重命名", "读取", "复制", "备份", "是否存在", "检查路径"}),
            examples=("读一下这个本地文件", "看下这个文件是否存在", "把这个文件名改成新的名字", "复制一份简历作为备份"),
        ),
    )


def _user_task(task: AgentTask) -> str:
    return str(task.input_payload.get("user_task") or task.goal or "").strip()


def _context_metadata(task: AgentTask, context: AgentRuntimeContext) -> dict[str, Any]:
    payload_context = task.input_payload.get("context_metadata")
    merged = dict(context.metadata)
    if isinstance(payload_context, dict):
        merged.update(payload_context)
    return merged


def _requested_operation(input_payload: dict[str, Any]) -> str | None:
    operation = str(input_payload.get("operation") or "").strip()
    if get_filesystem_operation_spec(operation) is not None:
        return operation
    return None


def _active_file_path(context_metadata: dict[str, Any]) -> str:
    active_file = context_metadata.get("active_file")
    if isinstance(active_file, dict):
        return str(active_file.get("path") or "").strip()
    return ""


def _path_from_task_context_or_text(*, task: AgentTask, user_task: str, context_metadata: dict[str, Any]) -> str:
    path = str(task.input_payload.get("path") or _active_file_path(context_metadata) or "").strip()
    if path:
        return path
    paths = extract_local_file_references(user_task)
    return paths[-1] if paths else ""


def _approval_payload(spec: FilesystemOperationSpec | None, arguments: dict[str, Any]) -> dict[str, Any]:
    if spec is None:
        return {"capability": FILESYSTEM_SKILL_CAPABILITY, "operation": "unknown", "requires_confirmation": True}
    return build_operation_approval_payload(spec, arguments)


def _approval_summary(spec: FilesystemOperationSpec | None, payload: dict[str, Any]) -> str:
    if spec is None:
        return "需要确认文件系统操作。"
    return format_operation_approval_summary(spec, payload)


def _execution_summary(spec: FilesystemOperationSpec, raw_payload: dict[str, Any], *, ok: bool) -> str:
    return format_operation_execution_summary(spec, raw_payload, ok=ok)


def _filesystem_trace_before_execution(spec: FilesystemOperationSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    source_path = _source_path_for_operation(spec, arguments)
    target_path = _target_path_for_operation(spec, arguments)
    precheck: dict[str, Any] = {}
    if source_path:
        precheck["source_path"] = source_path
        precheck["source_exists_before"] = _path_exists(source_path)
    if target_path:
        precheck["target_path"] = target_path
        precheck["target_exists_before"] = _path_exists(target_path)
    return {
        "operation": spec.operation,
        "goal_kind": spec.goal_kind,
        "precheck": precheck,
    }


def _filesystem_script_trace(spec: FilesystemOperationSpec, raw_payload: dict[str, Any]) -> dict[str, Any]:
    result = raw_payload.get("result") if isinstance(raw_payload.get("result"), dict) else {}
    return {
        "internal_tool": spec.legacy_tool_name,
        "internal_script": spec.script_name,
        "ok": bool(raw_payload.get("ok")),
        "error": raw_payload.get("error"),
        "return_code": result.get("return_code"),
        "stdout": _compact_text(result.get("stdout")),
        "stderr": _compact_text(result.get("stderr")),
    }


def _filesystem_postcheck(
    spec: FilesystemOperationSpec,
    arguments: dict[str, Any],
    *,
    raw_payload: dict[str, Any],
    ok: bool,
) -> dict[str, Any]:
    if spec.goal_kind == "file_to_file":
        return _file_to_file_postcheck(spec, arguments, ok=ok, raw_payload=raw_payload)
    if spec.goal_kind in {"path_exists", "read_content"}:
        path = str(arguments.get("path") or "").strip()
        return {
            "completed": ok,
            "path": path,
            "path_exists_after": _path_exists(path) if path else None,
            "reason": "script_ok" if ok else "script_failed",
        }
    return {"completed": ok, "reason": "script_ok" if ok else "script_failed"}


def _file_to_file_postcheck(
    spec: FilesystemOperationSpec,
    arguments: dict[str, Any],
    *,
    ok: bool,
    raw_payload: dict[str, Any],
) -> dict[str, Any]:
    source_path = _source_path_for_operation(spec, arguments)
    target_path = _target_path_for_operation(spec, arguments)
    source_exists_after = _path_exists(source_path) if source_path else None
    target_exists_after = _path_exists(target_path) if target_path else None
    if not ok:
        return {
            "completed": False,
            "source_path": source_path,
            "target_path": target_path,
            "source_exists_after": source_exists_after,
            "target_exists_after": target_exists_after,
            "reason": str(raw_payload.get("error") or "script_failed"),
        }

    same_path = bool(source_path and target_path and _same_path_text(source_path, target_path))
    if spec.operation == COPY_FILE_OPERATION:
        completed = bool(source_exists_after and target_exists_after)
        reason = "source_and_target_exist_after_copy" if completed else "copy_target_or_source_missing_after_script_success"
    elif spec.operation == RENAME_FILE_OPERATION:
        completed = bool(target_exists_after and (same_path or not source_exists_after))
        if completed:
            reason = "target_exists_and_source_removed_after_rename"
        elif not target_exists_after:
            reason = "target_missing_after_rename"
        else:
            reason = "source_still_exists_after_rename"
    else:
        requires_source_removed = "source_path_moved_or_renamed" in spec.success_criteria
        completed = bool(target_exists_after and (not requires_source_removed or same_path or not source_exists_after))
        reason = "file_to_file_postcheck_passed" if completed else "file_to_file_postcheck_failed"

    return {
        "completed": completed,
        "source_path": source_path,
        "target_path": target_path,
        "source_exists_after": source_exists_after,
        "target_exists_after": target_exists_after,
        "reason": reason,
    }


def _filesystem_postcheck_error(spec: FilesystemOperationSpec, postcheck: dict[str, Any]) -> str:
    reason = str(postcheck.get("reason") or "postcheck_failed")
    source = str(postcheck.get("source_path") or "").strip()
    target = str(postcheck.get("target_path") or "").strip()
    path_text = f" {source} -> {target}" if source and target else ""
    return f"{FILESYSTEM_POSTCHECK_ERROR_CODE}: {spec.summary}脚本返回成功，但运行时复核失败：{reason}{path_text}。"


def _source_path_for_operation(spec: FilesystemOperationSpec, arguments: dict[str, Any]) -> str:
    if "src" in spec.required_args or "src" in arguments:
        return str(arguments.get("src") or arguments.get("path") or "").strip()
    return str(arguments.get("path") or "").strip()


def _target_path_for_operation(spec: FilesystemOperationSpec, arguments: dict[str, Any]) -> str:
    target_key = spec.result_path_arg or "dst"
    return str(arguments.get(target_key) or arguments.get("dst") or "").strip()


def _path_exists(path: str) -> bool:
    try:
        return Path(path).exists()
    except (OSError, ValueError):
        return False


def _same_path_text(left: str, right: str) -> bool:
    return left.replace("\\", "/").rstrip("/").lower() == right.replace("\\", "/").rstrip("/").lower()


def _compact_text(value: Any, *, limit: int = 1000) -> str:
    text = str(value or "")
    return text if len(text) <= limit else f"{text[:limit]}..."


__all__ = ["FilesystemSkillExecutor", "classify_filesystem_operation"]
