from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import json
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
from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
from app.agent_runtime.web_search_query import normalize_external_web_search_query


@dataclass(frozen=True)
class LoopAgentTask:
    user_message: str
    available_capabilities: tuple[str, ...] = ()
    source_type: str = "agent_chat"
    context: dict[str, Any] = field(default_factory=dict)


class ToolChoiceLoopRunner:
    """Pi-Agent style inner loop that lets the model choose from offered tools."""

    def __init__(
        self,
        *,
        registry: AgentToolRegistry,
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
        bundle = _build_tool_schema_bundle(self._registry, task.available_capabilities)
        messages: list[dict[str, Any]] = _initial_messages(task)

        def decide_next_step(_trace) -> LoopAgentDecision:
            completion = self._llm_client.complete(
                messages=messages,
                tools=bundle["tools"] or None,
                tool_choice="auto" if bundle["tools"] else None,
            )
            tool_calls = list(getattr(completion, "tool_calls", []) or [])
            if not tool_calls:
                textual_decision = _textual_tool_call_decision(
                    str(getattr(completion, "content", "") or ""),
                    bundle=bundle,
                    task=task,
                )
                if textual_decision is not None:
                    return textual_decision
                return LoopAgentDecision(
                    action=LoopAgentAction.FINAL_ANSWER,
                    message=str(getattr(completion, "content", "") or "").strip(),
                    reason="模型认为当前信息已经足够，可以直接回答。",
                )

            tool_call = tool_calls[0]
            tool_alias = str(getattr(tool_call, "name", "") or "")
            requested_tool_name = bundle["alias_to_tool_name"].get(tool_alias, tool_alias)
            tool_input = _repair_relative_time_tool_input(
                task,
                requested_tool_name,
                dict(getattr(tool_call, "arguments", {}) or {}),
            )
            return LoopAgentDecision(
                action=LoopAgentAction.CALL_TOOL,
                capability=requested_tool_name,
                tool_input=tool_input,
                reason="模型判断当前任务需要调用工具。",
                metadata={
                    "tool_call_id": str(getattr(tool_call, "id", "") or ""),
                    "tool_alias": tool_alias,
                },
            )

        def execute_step(decision: LoopAgentDecision) -> LoopAgentObservation:
            if self._execute_tool_override is not None:
                observation = self._execute_tool_override(task, decision)
            else:
                observation = self._execute_tool(task, decision)
            messages.append(
                {
                    "role": "tool",
                    "content": _observation_message(decision, observation),
                }
            )
            return observation

        return LoopAgentController(max_steps=max_steps).run(
            decide_next_step=decide_next_step,
            execute_step=execute_step,
            session_id=session_id,
            task_id=task_id,
            run_id=run_id,
            event_sink=event_sink,
        )

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
                "不要把 Tool call 写成普通文本；需要工具时必须使用结构化工具调用。"
            ),
        }
    ]
    if task.context:
        messages.append({"role": "system", "content": f"本轮上下文：{task.context}"})
    messages.append({"role": "user", "content": task.user_message})
    return messages


def _build_tool_schema_bundle(registry: AgentToolRegistry, capabilities: tuple[str, ...]) -> dict[str, Any]:
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


_TEXTUAL_TOOL_CALL_RE = re.compile(
    r"^\s*(?:(?:\*\*)?OfferMaster\s+AI(?:\*\*)?\s*)?"
    r"(?:Tool\s*call|工具调用)\s*[:：]\s*"
    r"(?P<tool>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s*"
    r"(?P<arguments>\{.*\})\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _textual_tool_call_decision(content: str, *, bundle: dict[str, Any], task: LoopAgentTask) -> LoopAgentDecision | None:
    normalized = str(content or "").replace("\\_", "_").strip()
    if not normalized:
        return None
    for candidate in _textual_tool_call_candidates(normalized):
        match = _TEXTUAL_TOOL_CALL_RE.match(candidate)
        if match is None:
            continue
        raw_tool_name = match.group("tool")
        arguments = _parse_tool_call_arguments(match.group("arguments"))
        if arguments is None:
            continue
        requested_tool_name = bundle["alias_to_tool_name"].get(raw_tool_name, raw_tool_name)
        arguments = _repair_relative_time_tool_input(task, requested_tool_name, arguments)
        return LoopAgentDecision(
            action=LoopAgentAction.CALL_TOOL,
            capability=requested_tool_name,
            tool_input=arguments,
            reason="模型把工具调用写成了普通文本，运行时将其转换为真实工具调用。",
            metadata={"tool_call_id": "textual-tool-call", "tool_alias": raw_tool_name, "textual_tool_call": True},
        )
    return None


def _textual_tool_call_candidates(content: str) -> list[str]:
    candidates = [content]
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    for index, line in enumerate(lines):
        if "tool call" not in line.lower() and "工具调用" not in line:
            continue
        candidates.append(line)
        if index > 0 and "OfferMaster" in lines[index - 1]:
            candidates.append(f"{lines[index - 1]}\n{line}")
    return candidates


def _parse_tool_call_arguments(raw_arguments: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw_arguments)
    except Exception:
        return None
    return dict(parsed) if isinstance(parsed, dict) else None


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
    return (
        f"工具 {decision.capability} 执行状态：{observation.status}\n"
        f"观察结果：{observation.summary}\n"
        f"结构化结果：{observation.result_payload}"
    )


__all__ = ["LoopAgentTask", "ToolChoiceLoopRunner"]
