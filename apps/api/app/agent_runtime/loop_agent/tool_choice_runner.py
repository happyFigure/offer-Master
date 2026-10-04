from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
import logging
import re
from typing import Any, Callable

from app.agent_runtime.loop_agent.controller import LoopAgentController
from app.agent_runtime.loop_agent.events import LoopAgentEvent
from app.agent_runtime.loop_agent.schemas import (
    LoopAgentAction,
    LoopAgentDecision,
    LoopAgentObservation,
    LoopAgentRunResult,
)
from app.agent_runtime.tool_registry import AgentToolDefinition
from app.agent_runtime.web_search_query import normalize_external_web_search_query


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoopAgentStageContext:
    stage_id: str | None = None
    title: str | None = None
    objective: str | None = None
    status: str | None = None
    capability: str | None = None
    business_action: str | None = None
    allowed_capabilities: tuple[str, ...] = ()
    tool_strategy: dict[str, Any] = field(default_factory=dict)
    ranking_policy: tuple[str, ...] = ()
    received_context: dict[str, Any] = field(default_factory=dict)
    handoff_payload: dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "title": self.title,
            "objective": self.objective,
            "status": self.status,
            "capability": self.capability,
            "business_action": self.business_action,
            "allowed_capabilities": list(self.allowed_capabilities),
            "tool_strategy": dict(self.tool_strategy),
            "ranking_policy": list(self.ranking_policy),
            "received_context": dict(self.received_context),
            "handoff_payload": dict(self.handoff_payload),
        }


@dataclass(frozen=True)
class LoopAgentTask:
    user_message: str
    available_capabilities: tuple[str, ...] = ()
    source_type: str = "agent_chat"
    context: dict[str, Any] = field(default_factory=dict)
    stage_context: LoopAgentStageContext | dict[str, Any] | None = None


class ToolChoiceLoopRunner:
    """Pi-Agent style inner loop that lets the model choose from offered tools."""

    def __init__(
        self,
        *,
        registry: Any,
        llm_client: Any,
        db_session: Any | None = None,
        execute_tool: Callable[[LoopAgentTask, LoopAgentDecision], LoopAgentObservation] | None = None,
    ) -> None:
        self._registry = registry
        self._llm_client = llm_client
        self._db_session = db_session
        self._execute_tool_override = execute_tool

    def run(
        self,
        task: LoopAgentTask,
        *,
        max_steps: int,
        session_id: str | None = None,
        task_id: str | None = None,
        run_id: str | None = None,
        event_sink: Any | None = None,
    ) -> LoopAgentRunResult:
        all_capabilities = tuple(task.available_capabilities)
        all_alias_to_tool_name = _build_tool_schema_bundle(self._registry, all_capabilities)["alias_to_tool_name"]
        active_stage_context = _stage_context_metadata(_task_stage_context(task))
        stage_context_history = _stage_context_history(active_stage_context)
        stage_capabilities_history: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = _initial_messages(task)
        final_answer_guard_retries = 0
        model_turns = 0
        filesystem_recovery_observed = False
        filesystem_read_observed = False
        filesystem_mutation_completed = False
        filesystem_guard_prompt_injected = False

        def decide_next_step(_trace) -> LoopAgentDecision:
            nonlocal final_answer_guard_retries, model_turns
            current_capabilities = _stage_filtered_capabilities(active_stage_context, all_capabilities)
            _append_stage_capabilities_history(stage_capabilities_history, active_stage_context, current_capabilities)
            bundle = _build_tool_schema_bundle(self._registry, current_capabilities)
            while True:
                model_turns += 1
                logger.info(
                    "Tool choice model turn: turn=%s capabilities=%s message_count=%s guard_retries=%s",
                    model_turns,
                    current_capabilities,
                    len(messages),
                    final_answer_guard_retries,
                )
                completion = self._llm_client.complete(
                    messages=messages,
                    tools=bundle["tools"] or None,
                    tool_choice="auto" if bundle["tools"] else None,
                )
                tool_calls = list(getattr(completion, "tool_calls", []) or [])
                if tool_calls:
                    break

                final_message = str(getattr(completion, "content", "") or "").strip()
                guard_reason = _filesystem_final_answer_guard_reason(task, _trace)
                if (
                    guard_reason is None
                    and filesystem_recovery_observed
                    and filesystem_read_observed
                    and not filesystem_mutation_completed
                ):
                    guard_reason = (
                        "当前文件任务的 runtime 恢复信号要求读取后继续完成写操作，"
                        "不能把 read_file 当成最终结果。"
                    )
                if guard_reason and final_answer_guard_retries < 2:
                    final_answer_guard_retries += 1
                    # This is a runtime-owned continuation signal. It does not
                    # manufacture an operation or filename; it only prevents a
                    # prose answer from terminating an unfinished write flow.
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "RUNTIME_COMPLETION_GUARD: "
                                "运行时阻止了本轮提前结束。"
                                f"{guard_reason}"
                                "请不要返回普通文本，也不要等待用户确认；请根据已经读取到的真实内容，"
                                "提交 skill.filesystem 的结构化写操作调用。"
                                "由模型自己选择 copy_file 或 rename_file、目标路径和文件名意图；"
                                "runtime 只会校验、审批和执行你提交的结构化参数。"
                            ),
                            "metadata": {"source": "runtime_completion_guard", "retry": final_answer_guard_retries},
                        }
                    )
                    logger.info(
                        "Filesystem completion guard prompt appended: retry=%s message_count=%s",
                        final_answer_guard_retries,
                        len(messages),
                    )
                    continue
                if guard_reason:
                    return LoopAgentDecision(
                        action=LoopAgentAction.WAIT_USER,
                        message="文件任务尚未完成：模型还没有提交真实的文件写操作，runtime 已阻止提前结束。",
                        reason=guard_reason,
                        metadata={
                            "source": "runtime_completion_guard",
                            "guard_retries": final_answer_guard_retries,
                        },
                    )
                return LoopAgentDecision(
                    action=LoopAgentAction.FINAL_ANSWER,
                    message=final_message,
                    reason="模型认为当前信息已经足够，可以直接回答。",
                )

            tool_call = tool_calls[0]
            tool_alias = str(getattr(tool_call, "name", "") or "")
            requested_tool_name = bundle["alias_to_tool_name"].get(
                tool_alias,
                all_alias_to_tool_name.get(tool_alias, tool_alias),
            )
            tool_input = _repair_relative_time_tool_input(
                task,
                requested_tool_name,
                dict(getattr(tool_call, "arguments", {}) or {}),
            )
            return _decision_with_stage_policy_metadata(
                LoopAgentDecision(
                action=LoopAgentAction.CALL_TOOL,
                capability=requested_tool_name,
                tool_input=tool_input,
                reason="模型判断当前任务需要调用工具。",
                metadata={
                    "tool_call_id": str(getattr(tool_call, "id", "") or ""),
                    "tool_alias": tool_alias,
                },
                ),
                stage_context=active_stage_context,
                offered_capabilities=current_capabilities,
            )

        def execute_step(decision: LoopAgentDecision) -> LoopAgentObservation:
            nonlocal active_stage_context, stage_context_history
            nonlocal filesystem_recovery_observed, filesystem_read_observed, filesystem_mutation_completed, filesystem_guard_prompt_injected
            current_capabilities = _stage_filtered_capabilities(active_stage_context, all_capabilities)
            blocked_observation = _blocked_tool_decision_observation(task, decision, current_capabilities)
            if blocked_observation is not None:
                observation = blocked_observation
            elif self._execute_tool_override is not None:
                observation = self._execute_tool_override(task, decision)
            else:
                observation = self._execute_tool(task, decision)
            messages.append(
                {
                    "role": "tool",
                    "content": _observation_message(decision, observation),
                }
            )
            if str(decision.capability or "").strip() == "skill.filesystem":
                operation = str((decision.tool_input or {}).get("operation") or "").strip()
                observation_metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
                next_action = str(observation_metadata.get("next_action") or "").strip()
                error_code = str(observation_metadata.get("error_code") or "").strip()
                recovery_outcome = observation_metadata.get("recovery_outcome")
                recovery_next_action = (
                    str(recovery_outcome.get("next_action") or "").strip()
                    if isinstance(recovery_outcome, dict)
                    else ""
                )
                if next_action in {"continue_model_loop", "read_before_write"} or recovery_next_action in {"continue_model_loop", "read_before_write"} or error_code in {
                    "STRUCTURED_OPERATION_REQUIRED",
                    "CONTENT_BASED_NAME_REQUIRED",
                    "CONTENT_BASED_NAME_REQUIRES_READ",
                    "MISSING_REQUIRED_ARGUMENT",
                }:
                    filesystem_recovery_observed = True
                if operation == "read_file" and str(observation.status or "").strip().lower() in {"succeeded", "partial", "completed", "success", "ok"}:
                    filesystem_read_observed = True
                if operation in {"copy_file", "rename_file"} and str(observation.status or "").strip().lower() in {"succeeded", "completed", "success", "ok"}:
                    filesystem_mutation_completed = True
                logger.info(
                    "Filesystem loop evidence updated: operation=%s status=%s recovery=%s read=%s mutation=%s",
                    operation,
                    observation.status,
                    filesystem_recovery_observed,
                    filesystem_read_observed,
                    filesystem_mutation_completed,
                )
                if (
                    operation == "read_file"
                    and filesystem_recovery_observed
                    and filesystem_read_observed
                    and not filesystem_mutation_completed
                    and not filesystem_guard_prompt_injected
                ):
                    logger.info("Injecting filesystem completion guard after verified coarse read")
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "RUNTIME_COMPLETION_GUARD: 运行时阻止了本轮提前结束。"
                                "当前文件任务已经读取了源文件，下一步必须继续提交 skill.filesystem 的结构化写操作调用；"
                                "请由模型根据真实文件内容自行选择 copy_file 或 rename_file、目标路径和文件名意图，"
                                "不要返回普通文本，也不要等待用户确认。runtime 只负责校验、审批和执行。"
                            ),
                            "metadata": {"source": "runtime_completion_guard", "phase": "after_verified_read"},
                        }
                    )
                    logger.info(
                        "Filesystem completion guard prompt appended after read: message_count=%s",
                        len(messages),
                    )
                    filesystem_guard_prompt_injected = True
            transition_history = _stage_context_history_after_observation(active_stage_context, observation, decision)
            for stage_identifier in transition_history:
                stage_context_history = _append_unique_strings(stage_context_history, stage_identifier)
            next_stage_context = _next_stage_context_after_observation(active_stage_context, observation, decision)
            if next_stage_context:
                active_stage_context = next_stage_context
                stage_prompt = _stage_context_prompt(active_stage_context)
                if stage_prompt:
                    messages.append({"role": "system", "content": stage_prompt})
            return observation

        result = LoopAgentController(max_steps=max_steps).run(
            decide_next_step=decide_next_step,
            execute_step=execute_step,
            session_id=session_id,
            task_id=task_id,
            run_id=run_id,
            event_sink=event_sink,
        )
        if active_stage_context:
            result = replace(
                result,
                metadata={
                    **dict(result.metadata),
                    "stage_context": active_stage_context,
                    "stage_context_history": stage_context_history,
                    "stage_capabilities_history": stage_capabilities_history,
                },
            )
        return result

    def _execute_tool(self, task: LoopAgentTask, decision: LoopAgentDecision) -> LoopAgentObservation:
        tool_name = decision.capability or ""
        if tool_name not in task.available_capabilities:
            return LoopAgentObservation(
                status="failed",
                summary=f"模型请求了本轮未开放的工具：{tool_name}",
                tool_call_id=_decision_tool_call_id(decision),
                metadata={"error_code": "TOOL_NOT_OFFERED"},
            )

        definition = self._registry.get(tool_name)
        if definition is None:
            return LoopAgentObservation(
                status="failed",
                summary=f"模型请求了未注册的工具：{tool_name}",
                tool_call_id=_decision_tool_call_id(decision),
                metadata={"error_code": "TOOL_NOT_REGISTERED"},
            )
        if definition.handler is None:
            return LoopAgentObservation(
                status="failed",
                summary=f"工具没有可执行处理器：{tool_name}",
                tool_call_id=_decision_tool_call_id(decision),
                metadata={"error_code": "TOOL_HANDLER_MISSING"},
            )

        missing_required = _missing_required_tool_input_names(definition, decision.tool_input)
        if missing_required:
            if _is_model_recoverable_input_error(definition):
                return LoopAgentObservation(
                    status="failed",
                    summary=f"工具参数还不完整：缺少 {', '.join(missing_required)}。请根据这个结构化错误继续补全工具参数。",
                    tool_call_id=_decision_tool_call_id(decision),
                    metadata={
                        "error_code": "TOOL_INPUT_INVALID",
                        "tool_input": dict(decision.tool_input),
                        "missing_required_fields": missing_required,
                        "recoverable": True,
                        "next_action": "continue_model_loop",
                    },
                )
            return LoopAgentObservation(
                status="waiting_user",
                summary=f"工具参数还不完整：缺少 {', '.join(missing_required)}。请补充后我再继续。",
                requires_user_action=True,
                tool_call_id=_decision_tool_call_id(decision),
                metadata={
                    "error_code": "TOOL_INPUT_INVALID",
                    "tool_input": dict(decision.tool_input),
                    "missing_required_fields": missing_required,
                },
            )

        validation_error = _validate_tool_input(definition, decision.tool_input)
        if validation_error is not None:
            return LoopAgentObservation(
                status="failed",
                summary=validation_error,
                tool_call_id=_decision_tool_call_id(decision),
                metadata={"error_code": "TOOL_INPUT_INVALID", "tool_input": dict(decision.tool_input)},
            )

        payload = definition.handler(self._db_session, **decision.tool_input)
        ok = _payload_ok(payload)
        quality_observation = _tool_result_quality_observation(decision, payload, ok=ok)
        if quality_observation is not None:
            return quality_observation
        return LoopAgentObservation(
            status="succeeded" if ok else "failed",
            summary=_summarize_tool_payload(payload),
            result_payload=_payload_to_dict(payload),
            tool_call_id=_decision_tool_call_id(decision),
        )


def _initial_messages(task: LoopAgentTask) -> list[dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": (
                "你是 OfferMaster 主 agent。你可以直接回答，也可以从本轮提供的工具里选择一个调用。"
                "如果工具结果不够，可以继续判断下一步；如果信息足够，就输出最终答案。"
                f"当前日期是 {date.today().isoformat()}。"
                "如果用户说今天、本周、这周、这个星期、最近、最新等相对时间，必须基于当前日期生成查询，"
                "不要使用过期年份。"
                "如果用户要求把本地文件中的一段文本精确替换成另一段，并强调其他不要动，"
                "优先使用精确替换工具，不要让模型重写整个文件。"
                "不要把 Tool call 写成普通文本；需要工具时必须使用结构化工具调用。"
            ),
        }
    ]
    stage_prompt = _stage_context_prompt(_task_stage_context(task))
    if stage_prompt:
        messages.append({"role": "system", "content": stage_prompt})
    if task.context:
        messages.append({"role": "system", "content": f"本轮上下文：{task.context}"})
    messages.append({"role": "user", "content": task.user_message})
    return messages


def _filesystem_final_answer_guard_reason(
    task: LoopAgentTask,
    trace: list[Any],
) -> str | None:
    """Return a structured reason when a filesystem write is still pending."""

    contract = task.context.get("filesystem_completion_contract") if isinstance(task.context, dict) else None
    contract_active = isinstance(contract, dict) and bool(
        contract.get("awaiting_write") or contract.get("requires_follow_up_after_read")
    )
    recovery_observed = False
    read_observed = False

    logger.warning(
        "Filesystem completion guard inspect: contract=%s trace=%s",
        contract,
        [
            {
                "operation": getattr(entry, "metadata", {}).get("observation", {}).get("metadata", {}).get("tool_input", {}).get("operation")
                if isinstance(getattr(entry, "metadata", {}).get("observation", {}), dict)
                else None,
                "status": getattr(entry, "observation_status", None),
                "metadata": getattr(entry, "metadata", {}).get("observation", {}).get("metadata", {})
                if isinstance(getattr(entry, "metadata", {}).get("observation", {}), dict)
                else {},
            }
            for entry in trace
            if str(getattr(entry, "capability", "") or "").strip() == "skill.filesystem"
        ],
    )

    for entry in trace:
        capability = str(getattr(entry, "capability", "") or "").strip()
        if capability != "skill.filesystem":
            continue
        observation = getattr(entry, "metadata", {}).get("observation", {})
        observation_metadata = observation.get("metadata") if isinstance(observation, dict) else {}
        observation_metadata = observation_metadata if isinstance(observation_metadata, dict) else {}
        tool_input = observation_metadata.get("tool_input") if isinstance(observation_metadata.get("tool_input"), dict) else {}
        operation = str(tool_input.get("operation") or "").strip()
        observation_status = str(getattr(entry, "observation_status", "") or "").strip().lower()
        if operation == "read_file" and observation_status in {"succeeded", "partial", "completed", "success", "ok"}:
            read_observed = True
        if operation in {"copy_file", "rename_file"} and observation_status in {"succeeded", "completed", "success", "ok"}:
            return None
        if str(observation_metadata.get("next_action") or "").strip() in {"continue_model_loop", "read_before_write"}:
            recovery_observed = True
        recovery_outcome = observation_metadata.get("recovery_outcome")
        if isinstance(recovery_outcome, dict) and str(recovery_outcome.get("next_action") or "").strip() in {"continue_model_loop", "read_before_write"}:
            recovery_observed = True
        if observation_metadata.get("error_code") in {
            "STRUCTURED_OPERATION_REQUIRED",
            "CONTENT_BASED_NAME_REQUIRED",
            "CONTENT_BASED_NAME_REQUIRES_READ",
            "MISSING_REQUIRED_ARGUMENT",
        }:
            recovery_observed = True
        # A missing structured operation is itself a runtime-owned recovery
        # signal. Do not parse the user's prose here; only use the native
        # tool input and the structured observation emitted by the executor.
        if not operation and observation_status in {"failed", "partial", "completed"}:
            if observation_metadata.get("recoverable") is True or observation_metadata.get("error_code"):
                recovery_observed = True

    if (
        contract_active
        and (bool(contract.get("awaiting_write")) or read_observed)
        and not _trace_contains_completed_filesystem_write(trace)
    ):
        return "当前文件任务已经读取了源文件，但还没有完成模型选择的 copy_file 或 rename_file 写操作。"
    if recovery_observed and read_observed and not _trace_contains_completed_filesystem_write(trace):
        return "当前文件任务的 runtime 恢复信号要求读取后继续完成写操作，不能把 read_file 当成最终结果。"
    return None


def _trace_contains_completed_filesystem_write(trace: list[Any]) -> bool:
    for entry in trace:
        if str(getattr(entry, "capability", "") or "").strip() != "skill.filesystem":
            continue
        observation = getattr(entry, "metadata", {}).get("observation", {})
        observation_metadata = observation.get("metadata") if isinstance(observation, dict) else {}
        observation_metadata = observation_metadata if isinstance(observation_metadata, dict) else {}
        tool_input = observation_metadata.get("tool_input") if isinstance(observation_metadata.get("tool_input"), dict) else {}
        operation = str(tool_input.get("operation") or "").strip()
        observation_status = str(getattr(entry, "observation_status", "") or "").strip().lower()
        if operation in {"copy_file", "rename_file"} and observation_status in {"succeeded", "completed", "success", "ok"}:
            return True
    return False


def _first_nested_runtime_field(payload: dict[str, Any] | None, key: str) -> Any | None:
    """Find one runtime-owned field without interpreting user prose."""

    if not isinstance(payload, dict):
        return None
    queue: list[Any] = [payload]
    visited: set[int] = set()
    while queue:
        current = queue.pop(0)
        if not isinstance(current, dict) or id(current) in visited:
            continue
        visited.add(id(current))
        if key in current and current[key] is not None:
            return current[key]
        for child_key in ("result", "raw_result", "result_envelope", "error_details", "output_payload"):
            child = current.get(child_key)
            if isinstance(child, dict):
                queue.append(child)
    return None


def _task_stage_context(task: LoopAgentTask) -> LoopAgentStageContext | dict[str, Any] | None:
    if task.stage_context is not None:
        return task.stage_context
    candidate = task.context.get("stage_context") if isinstance(task.context, dict) else None
    return candidate if isinstance(candidate, dict) else None


def _stage_context_metadata(stage_context: LoopAgentStageContext | dict[str, Any] | None) -> dict[str, Any]:
    if stage_context is None:
        return {}
    if isinstance(stage_context, LoopAgentStageContext):
        return {key: value for key, value in stage_context.to_metadata_dict().items() if value not in (None, {}, [])}
    if isinstance(stage_context, dict):
        return {str(key): value for key, value in stage_context.items() if value not in (None, {}, [])}
    return {}


def _stage_context_prompt(stage_context: LoopAgentStageContext | dict[str, Any] | None) -> str:
    metadata = _stage_context_metadata(stage_context)
    if not metadata:
        return ""

    lines = ["当前是一个多阶段 agent 任务。做下一步判断时，要优先复用上游阶段已经交接的信息，不要重复完成已完成阶段。"]
    title = str(metadata.get("title") or metadata.get("stage_title") or "").strip()
    if title:
        lines.append(f"当前任务阶段：{title}")
    stage_id = str(metadata.get("stage_id") or "").strip()
    if stage_id:
        lines.append(f"阶段标识：{stage_id}")
    objective = str(metadata.get("objective") or "").strip()
    if objective:
        lines.append(f"阶段目标：{objective}")
    status = str(metadata.get("status") or metadata.get("execution_status") or "").strip()
    if status:
        lines.append(f"阶段状态：{status}")
    capability = str(metadata.get("capability") or "").strip()
    if capability:
        lines.append(f"阶段能力：{capability}")
    business_action = str(metadata.get("business_action") or "").strip()
    if business_action:
        lines.append(f"本阶段业务动作：{business_action}")
    tool_strategy = _stage_tool_strategy(metadata)
    allowed_capabilities = _stage_allowed_capabilities(metadata)
    strategy_mode = str(tool_strategy.get("mode") or "").strip()
    if strategy_mode == "none":
        lines.append("本阶段工具策略：不调用工具，只基于已有信息分析或整理。")
    elif allowed_capabilities:
        lines.append(f"本阶段可选工具：{', '.join(allowed_capabilities)}")
    strategy_description = str(tool_strategy.get("description") or "").strip()
    if strategy_description:
        lines.append(f"工具选择说明：{strategy_description}")
    ranking_policy = _string_list(metadata.get("ranking_policy"))
    if ranking_policy:
        lines.append("匹配排序规则：")
        for item in ranking_policy:
            lines.append(f"- {item}")

    received_context = metadata.get("received_context")
    if isinstance(received_context, dict) and received_context:
        lines.append("上游阶段交接信息：")
        summary = str(received_context.get("summary") or "").strip()
        if summary:
            lines.append(f"- 摘要：{summary}")
        tool_names = _string_list(received_context.get("tool_names"))
        if tool_names:
            lines.append(f"- 上游工具：{', '.join(tool_names)}")
        upstream_stage_ids = _string_list(received_context.get("upstream_stage_ids"))
        if upstream_stage_ids:
            lines.append(f"- 上游阶段：{', '.join(upstream_stage_ids)}")

    handoff_payload = metadata.get("handoff_payload")
    if isinstance(handoff_payload, dict) and handoff_payload:
        lines.append("本阶段已产生的阶段产物：")
        summary = str(handoff_payload.get("summary") or "").strip()
        if summary:
            lines.append(f"- 摘要：{summary}")
        tool_names = _string_list(handoff_payload.get("tool_names"))
        if tool_names:
            lines.append(f"- 已用工具：{', '.join(tool_names)}")
    return "\n".join(lines)


def _next_stage_context_after_observation(
    current_stage_context: dict[str, Any],
    observation: LoopAgentObservation,
    decision: LoopAgentDecision,
) -> dict[str, Any]:
    next_stage_index = _next_stage_index_after_observation(current_stage_context, observation, decision)
    if next_stage_index is None:
        return {}
    plan = _stage_context_plan(current_stage_context)
    if next_stage_index < 0 or next_stage_index >= len(plan):
        return {}
    next_stage = dict(plan[next_stage_index])
    next_stage.setdefault("stage_plan", plan)
    next_stage["status"] = "running"
    return next_stage


def _stage_context_history_after_observation(
    current_stage_context: dict[str, Any],
    observation: LoopAgentObservation,
    decision: LoopAgentDecision,
) -> list[str]:
    next_stage_index = _next_stage_index_after_observation(current_stage_context, observation, decision)
    if next_stage_index is None:
        return []
    plan = _stage_context_plan(current_stage_context)
    current_identifier = _stage_context_identifier(current_stage_context)
    current_index = _stage_context_plan_index(plan, current_identifier)
    if current_index is None:
        current_index = -1
    return [
        _stage_context_identifier(stage)
        for stage in plan[current_index + 1 : next_stage_index + 1]
        if _stage_context_identifier(stage)
    ]


def _next_stage_index_after_observation(
    current_stage_context: dict[str, Any],
    observation: LoopAgentObservation,
    decision: LoopAgentDecision,
) -> int | None:
    if not current_stage_context:
        return None
    if observation.status != "succeeded" or observation.requires_user_action or observation.suggested_next_decision is not None:
        return None
    # Filesystem operations form one model-directed transaction: read_file is
    # commonly followed by copy_file or rename_file after the model inspects
    # the returned content.  Do not advance an unrelated business stage here,
    # otherwise its allow-list can hide skill.filesystem before the write step.
    if str(decision.capability or "").strip() in {"skill.filesystem", "skill_filesystem"}:
        logger.info(
            "Keeping stage context after filesystem observation: capability=%s stage=%s",
            decision.capability,
            _stage_context_identifier(current_stage_context),
        )
        return None
    plan = _stage_context_plan(current_stage_context)
    if not plan:
        return None
    current_identifier = _stage_context_identifier(current_stage_context)
    if not current_identifier:
        return None
    current_index = _stage_context_plan_index(plan, current_identifier)
    if current_index is None:
        return None
    observed_stage_index = _stage_context_plan_index_for_capability(plan, decision.capability)
    if observed_stage_index is not None and observed_stage_index >= current_index:
        next_index = observed_stage_index + 1
    else:
        next_index = current_index + 1
    if next_index >= len(plan):
        return None
    return next_index


def _stage_context_plan(stage_context: dict[str, Any]) -> list[dict[str, Any]]:
    raw_plan = stage_context.get("stage_plan")
    if not isinstance(raw_plan, list):
        return []
    plan: list[dict[str, Any]] = []
    for item in raw_plan:
        metadata = _stage_context_metadata(item) if isinstance(item, (dict, LoopAgentStageContext)) else {}
        if metadata:
            plan.append(metadata)
    return plan


def _stage_context_plan_index(plan: list[dict[str, Any]], identifier: str) -> int | None:
    for index, stage in enumerate(plan):
        if _stage_context_identifier(stage) == identifier:
            return index
    return None


def _stage_context_plan_index_for_capability(plan: list[dict[str, Any]], capability: str | None) -> int | None:
    capability_name = str(capability or "").strip()
    if not capability_name:
        return None
    for index, stage in enumerate(plan):
        if capability_name in _stage_allowed_capabilities(stage):
            return index
    return None


def _stage_context_history(stage_context: dict[str, Any]) -> list[str]:
    identifier = _stage_context_identifier(stage_context)
    return [identifier] if identifier else []


def _stage_context_identifier(stage_context: dict[str, Any]) -> str:
    return str(stage_context.get("stage_id") or stage_context.get("capability") or stage_context.get("title") or "").strip()


def _stage_filtered_capabilities(stage_context: dict[str, Any], available_capabilities: tuple[str, ...]) -> tuple[str, ...]:
    if not stage_context:
        return available_capabilities
    tool_strategy = _stage_tool_strategy(stage_context)
    if str(tool_strategy.get("mode") or "").strip() == "none":
        return ()
    allowed_capabilities = _stage_allowed_capabilities(stage_context)
    if not allowed_capabilities:
        return available_capabilities
    allowed = set(allowed_capabilities)
    return tuple(capability for capability in available_capabilities if capability in allowed)


def _stage_allowed_capabilities(stage_context: dict[str, Any]) -> tuple[str, ...]:
    return tuple(_string_list(stage_context.get("allowed_capabilities")))


def _stage_tool_strategy(stage_context: dict[str, Any]) -> dict[str, Any]:
    value = stage_context.get("tool_strategy")
    return dict(value) if isinstance(value, dict) else {}


def _append_stage_capabilities_history(
    history: list[dict[str, Any]],
    stage_context: dict[str, Any],
    capabilities: tuple[str, ...],
) -> None:
    stage_id = _stage_context_identifier(stage_context)
    snapshot = {"stage_id": stage_id, "capabilities": list(capabilities)}
    if history and history[-1] == snapshot:
        return
    history.append(snapshot)


def _decision_with_stage_policy_metadata(
    decision: LoopAgentDecision,
    *,
    stage_context: dict[str, Any],
    offered_capabilities: tuple[str, ...],
) -> LoopAgentDecision:
    if decision.action != LoopAgentAction.CALL_TOOL:
        return decision
    stage_id = _stage_context_identifier(stage_context)
    return replace(
        decision,
        metadata={
            **dict(decision.metadata),
            "stage_id": stage_id,
            "offered_capabilities": list(offered_capabilities),
        },
    )


def _blocked_tool_decision_observation(
    task: LoopAgentTask,
    decision: LoopAgentDecision,
    offered_capabilities: tuple[str, ...],
) -> LoopAgentObservation | None:
    if decision.action != LoopAgentAction.CALL_TOOL:
        return None
    tool_name = decision.capability or ""
    if tool_name not in task.available_capabilities:
        return LoopAgentObservation(
            status="failed",
            summary=f"模型请求了本轮未开放的工具：{tool_name}",
            tool_call_id=_decision_tool_call_id(decision),
            metadata={"error_code": "TOOL_NOT_OFFERED"},
        )
    if tool_name not in offered_capabilities:
        return LoopAgentObservation(
            status="failed",
            summary=f"模型请求了当前阶段不允许使用的工具：{tool_name}",
            tool_call_id=_decision_tool_call_id(decision),
            metadata={
                "error_code": "TOOL_NOT_ALLOWED_IN_STAGE",
                "offered_capabilities": list(offered_capabilities),
            },
        )
    return None


def _append_unique_strings(values: list[str], value: str) -> list[str]:
    if not value or value in values:
        return values
    return [*values, value]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _build_tool_schema_bundle(registry: Any, capabilities: tuple[str, ...]) -> dict[str, Any]:
    tools: list[dict[str, Any]] = []
    alias_to_tool_name: dict[str, str] = {}
    used_aliases: set[str] = set()
    for tool_name in capabilities:
        definition = registry.get(tool_name)
        if definition is None:
            continue
        alias = _safe_tool_alias(tool_name)
        base_alias = alias
        suffix = 2
        while alias in used_aliases:
            alias = f"{base_alias}_{suffix}"
            suffix += 1
        used_aliases.add(alias)
        alias_to_tool_name[alias] = tool_name
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": alias,
                    "description": definition.description,
                    "parameters": definition.input_schema,
                },
            }
        )
    return {"tools": tools, "alias_to_tool_name": alias_to_tool_name}


def _safe_tool_alias(tool_name: str) -> str:
    alias = re.sub(r"[^A-Za-z0-9_]", "_", tool_name).strip("_")
    if not alias:
        alias = "tool"
    if alias[0].isdigit():
        alias = f"tool_{alias}"
    return alias


def _validate_tool_input(definition: AgentToolDefinition, tool_input: dict[str, Any]) -> str | None:
    schema = definition.input_schema if isinstance(definition.input_schema, dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    missing = [str(name) for name in required if str(name) not in tool_input]
    if missing:
        return f"工具参数缺失：{', '.join(missing)}"

    if schema.get("additionalProperties") is False:
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        unknown = sorted(set(tool_input) - set(properties))
        if unknown:
            return f"工具参数包含未声明字段：{', '.join(unknown)}"
    return None


def _missing_required_tool_input_names(definition: AgentToolDefinition, tool_input: dict[str, Any]) -> list[str]:
    schema = definition.input_schema if isinstance(definition.input_schema, dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    return [str(name) for name in required if str(name) not in tool_input]


def _is_model_recoverable_input_error(definition: AgentToolDefinition) -> bool:
    """Let the model repair filesystem arguments before asking the user."""

    if definition.name.startswith("filesystem."):
        return True
    profile = definition.candidate_profile
    return bool(profile and "filesystem_operation" in profile.categories)


_YEAR_RE = re.compile(r"20\d{2}")
_RELATIVE_TIME_MARKERS = (
    "今天",
    "现在",
    "最新",
    "最近",
    "明天",
    "本周",
    "这周",
    "这个星期",
    "这星期",
    "本星期",
    "这个礼拜",
    "本礼拜",
    "今晚",
    "this week",
    "today",
    "latest",
    "recent",
)


def _repair_relative_time_tool_input(task: LoopAgentTask, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    if tool_name != "external.web_search" or "query" not in tool_input:
        return tool_input
    user_message = str(task.user_message or "")
    query = str(tool_input.get("query") or "").strip()
    if not query:
        return tool_input
    if _mentions_relative_time(user_message):
        current_date = date.today().isoformat()
        user_years = set(_YEAR_RE.findall(user_message))
        query_years = set(_YEAR_RE.findall(query))
        if query_years - user_years:
            query = f"{user_message} {current_date} schedule"
        elif current_date not in query:
            query = f"{query} {current_date}"
    query = _normalize_external_web_search_query_aliases(query)
    return {**tool_input, "query": query}


def _mentions_relative_time(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _RELATIVE_TIME_MARKERS)


def _normalize_external_web_search_query_aliases(query: str) -> str:
    return normalize_external_web_search_query(query)


def _decision_tool_call_id(decision: LoopAgentDecision) -> str | None:
    raw = decision.metadata.get("tool_call_id") if isinstance(decision.metadata, dict) else None
    return str(raw) if raw else None


def _payload_ok(payload: Any) -> bool:
    if isinstance(payload, dict) and "ok" in payload:
        return bool(payload.get("ok"))
    if hasattr(payload, "ok"):
        return bool(payload.ok)
    return True


def _payload_to_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "to_metadata_dict"):
        value = payload.to_metadata_dict()
        return value if isinstance(value, dict) else {"value": value}
    return {"value": str(payload)}


def _tool_result_quality_observation(
    decision: LoopAgentDecision,
    payload: Any,
    *,
    ok: bool,
) -> LoopAgentObservation | None:
    if not ok:
        return None
    report = _garbled_text_report(payload)
    if report is None:
        return None

    metadata = {
        "quality_gate": "TOOL_RESULT_GARBLED",
        "garbled_text": report,
    }
    retry_decision = _garbled_text_retry_decision(decision)
    if retry_decision is not None:
        return LoopAgentObservation(
            status="partial",
            summary="工具结果疑似乱码，不能直接当作成功结果；运行时会换一种读取方式再试一次。",
            result_payload=_payload_to_dict(payload),
            tool_call_id=_decision_tool_call_id(decision),
            metadata=metadata,
            suggested_next_decision=retry_decision,
        )
    return LoopAgentObservation(
        status="failed",
        summary="工具结果疑似乱码，不能可靠回答用户问题。",
        result_payload=_payload_to_dict(payload),
        tool_call_id=_decision_tool_call_id(decision),
        metadata=metadata,
    )


def _garbled_text_retry_decision(decision: LoopAgentDecision) -> LoopAgentDecision | None:
    if decision.capability != "filesystem.read_file":
        return None
    current_encoding = str(decision.tool_input.get("encoding") or "auto").strip().lower()
    if current_encoding in {"utf-8", "utf8", "utf-8-sig"}:
        next_encoding = "gb18030"
    elif current_encoding in {"gb18030", "gbk", "cp936"}:
        return None
    else:
        next_encoding = "utf-8"
    return LoopAgentDecision(
        action=LoopAgentAction.CALL_TOOL,
        capability=decision.capability,
        tool_input={**decision.tool_input, "encoding": next_encoding},
        reason=f"上一次读取结果疑似乱码，改用 {next_encoding} 编码重新读取。",
        metadata={"source": "encoding_quality", "previous_encoding": current_encoding},
    )


def _garbled_text_report(payload: Any) -> dict[str, Any] | None:
    strings = _collect_payload_strings(_payload_to_dict(payload))
    if not strings:
        return None
    worst: dict[str, Any] | None = None
    for path, text in strings:
        report = _single_text_garbled_report(text)
        if report is None:
            continue
        candidate = {**report, "path": path}
        if worst is None or int(candidate["score"]) > int(worst["score"]):
            worst = candidate
    return worst


def _collect_payload_strings(value: Any, *, path: str = "payload", limit: int = 20) -> list[tuple[str, str]]:
    if limit <= 0:
        return []
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        strings: list[tuple[str, str]] = []
        for key, item in value.items():
            if len(strings) >= limit:
                break
            strings.extend(_collect_payload_strings(item, path=f"{path}.{key}", limit=limit - len(strings)))
        return strings[:limit]
    if isinstance(value, list):
        strings = []
        for index, item in enumerate(value[:limit]):
            if len(strings) >= limit:
                break
            strings.extend(_collect_payload_strings(item, path=f"{path}[{index}]", limit=limit - len(strings)))
        return strings[:limit]
    return []


def _single_text_garbled_report(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    sample = text[:4000]
    replacement_count = sample.count("�")
    mojibake_markers = sum(sample.count(marker) for marker in ("锟斤拷", "ï¿½", "Ã", "Â"))
    suspicious_control_count = sum(1 for char in sample if ord(char) < 32 and char not in "\r\n\t")
    score = replacement_count * 3 + mojibake_markers * 2 + suspicious_control_count
    ratio = (replacement_count + mojibake_markers + suspicious_control_count) / max(len(sample), 1)
    if replacement_count >= 3 or score >= 8 or ratio >= 0.03:
        return {
            "score": score,
            "replacement_char_count": replacement_count,
            "mojibake_marker_count": mojibake_markers,
            "control_char_count": suspicious_control_count,
            "sample": sample[:240],
        }
    return None


def _summarize_tool_payload(payload: Any) -> str:
    data = _payload_to_dict(payload)
    candidates: list[Any] = [data.get("summary"), data.get("message")]
    result = data.get("result")
    if isinstance(result, dict):
        candidates.extend([result.get("answer"), result.get("summary"), result.get("message"), result.get("status")])
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "工具已返回结果。" if _payload_ok(payload) else "工具执行失败。"


def _observation_message(decision: LoopAgentDecision, observation: LoopAgentObservation) -> str:
    message = (
        f"工具 {decision.capability} 执行状态：{observation.status}\n"
        f"观察结果：{observation.summary}\n"
        f"结构化结果：{observation.result_payload}\n"
        f"运行元数据：{observation.metadata}"
    )
    if str(decision.capability or "").strip() == "skill.filesystem" and str(
        (decision.tool_input or {}).get("operation") or ""
    ).strip() == "read_file":
        content = _filesystem_read_content_from_observation(observation.result_payload)
        if content:
            # Keep the model-facing evidence explicit. The raw nested payload is
            # retained for auditability, but a prose-only dict dump is not a
            # reliable interface for content-based follow-up decisions.
            logger.info("Filesystem read content promoted to model evidence: chars=%s", len(content))
            message += f"\n文件真实内容（可用于后续判断）：\n{content}"
    return message


def _filesystem_read_content_from_observation(payload: Any) -> str:
    """Extract read_file content from the runtime result without semantic parsing."""

    queue: list[Any] = [payload]
    visited: set[int] = set()
    while queue:
        current = queue.pop(0)
        if not isinstance(current, dict) or id(current) in visited:
            continue
        visited.add(id(current))
        content = current.get("content")
        if isinstance(content, str) and content.strip():
            return content[:12000]
        for child_key in ("result", "raw_result", "output_payload", "result_envelope"):
            child = current.get(child_key)
            if isinstance(child, dict):
                queue.append(child)
    return ""


__all__ = ["LoopAgentStageContext", "LoopAgentTask", "ToolChoiceLoopRunner"]
