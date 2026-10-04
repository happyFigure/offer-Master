from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any
from uuid import uuid4

from app.agent_runtime.agent_as_tool import AgentRuntimeContext
from app.agent_runtime.sdk_agents.sandbox import SandboxPathError, SdkAgentSandboxManager, SdkAgentSandboxWorkspace
from app.agent_runtime.tool_registry import (
    FILESYSTEM_COPY_FILE_TOOL,
    FILESYSTEM_DELETE_PATH_TOOL,
    FILESYSTEM_LIST_DIR_TOOL,
    FILESYSTEM_MAKE_DIR_TOOL,
    FILESYSTEM_MOVE_FILE_TOOL,
    FILESYSTEM_PATH_EXISTS_TOOL,
    FILESYSTEM_PATH_STAT_TOOL,
    FILESYSTEM_READ_FILE_TOOL,
    FILESYSTEM_REPLACE_TEXT_TOOL,
    FILESYSTEM_WRITE_TEXT_TOOL,
    AgentToolDefinition,
    AgentToolRegistry,
    AgentToolRiskLevel,
)


DEFAULT_SDK_TOOL_OBSERVATION_CHARS = 2_000
SDK_AGENT_SANDBOX_METADATA_KEY = "sdk_agent_sandbox"
SDK_AGENT_RUNTIME_MCP_CALLS_KEY = "_sdk_runtime_mcp_calls"
DBX_READONLY_CAPABILITY_ID = "agent.dbx_readonly"
DBX_CONNECTION_DISCOVERY_TOOL = "mcp.dbx.dbx_list_connections"

logger = logging.getLogger(__name__)

_SANDBOX_READ_PATH_TOOLS = frozenset(
    {
        FILESYSTEM_LIST_DIR_TOOL,
        FILESYSTEM_PATH_EXISTS_TOOL,
        FILESYSTEM_PATH_STAT_TOOL,
        FILESYSTEM_READ_FILE_TOOL,
    }
)
_SANDBOX_SINGLE_PATH_WRITE_TOOLS = {
    FILESYSTEM_WRITE_TEXT_TOOL: "write",
    FILESYSTEM_REPLACE_TEXT_TOOL: "replace",
    FILESYSTEM_DELETE_PATH_TOOL: "delete",
    FILESYSTEM_MAKE_DIR_TOOL: "mkdir",
}
_SANDBOX_FILE_TOOLS = _SANDBOX_READ_PATH_TOOLS | frozenset(_SANDBOX_SINGLE_PATH_WRITE_TOOLS) | frozenset(
    {FILESYSTEM_COPY_FILE_TOOL, FILESYSTEM_MOVE_FILE_TOOL}
)


@dataclass(frozen=True)
class SdkAgentToolGateway:
    tool_registry: AgentToolRegistry
    session_provider: Callable[[AgentRuntimeContext], Any] | None = None
    max_observation_chars: int = DEFAULT_SDK_TOOL_OBSERVATION_CHARS

    def build_tools(self, tool_names: Iterable[str], context: AgentRuntimeContext) -> list[Any]:
        try:
            from agents import FunctionTool
        except ImportError:
            return []

        tools: list[Any] = []
        for alias, tool_name in self.safe_aliases(tool_names).items():
            definition = self.tool_registry.get(tool_name)
            if definition is None or not self._can_delegate(definition, context):
                logger.debug(
                    "Skipped SDK child-agent tool registration",
                    extra={"tool_name": tool_name, "alias": alias, "run_id": context.run_id, "task_id": context.task_id},
                )
                continue
            tools.append(self._build_function_tool(FunctionTool, alias=alias, definition=definition, context=context))
        return tools

    @staticmethod
    def safe_aliases(tool_names: Iterable[str]) -> dict[str, str]:
        aliases: dict[str, str] = {}
        used: set[str] = set()
        for tool_name in tool_names:
            original = str(tool_name or "").strip()
            if not original:
                continue
            base_alias = _safe_tool_alias(original)
            alias = base_alias
            suffix = 2
            while alias in used:
                alias = f"{base_alias[:58]}_{suffix}"
                suffix += 1
            used.add(alias)
            aliases[alias] = original
        return aliases

    def _can_delegate(self, definition: AgentToolDefinition, context: AgentRuntimeContext) -> bool:
        if definition.handler is None:
            return False
        risk_value = str(getattr(definition.risk_level, "value", definition.risk_level)).strip().lower()
        is_mcp_tool = definition.name.startswith("mcp.")
        sandbox_allowed = self._can_delegate_with_sandbox(definition, context)
        if risk_value != AgentToolRiskLevel.LOW.value and not (sandbox_allowed or (is_mcp_tool and definition.requires_confirmation)):
            return False
        if definition.requires_confirmation and not (sandbox_allowed or is_mcp_tool):
            return False
        source_type = str(context.permission_scope.get("source_type") or "agent_chat").strip()
        if definition.allowed_source_types and source_type not in definition.allowed_source_types:
            return False
        return True

    def _can_delegate_with_sandbox(self, definition: AgentToolDefinition, context: AgentRuntimeContext) -> bool:
        return definition.name in _SANDBOX_FILE_TOOLS and _sandbox_workspace_from_context(context) is not None

    def _build_function_tool(
        self,
        function_tool_cls: type[Any],
        *,
        alias: str,
        definition: AgentToolDefinition,
        context: AgentRuntimeContext,
    ) -> Any:
        async def invoke_tool(_tool_context: Any, arguments_json: str) -> str:
            try:
                arguments = _parse_tool_arguments(arguments_json)
            except ValueError as exc:
                logger.info(
                    "SDK child-agent tool received invalid JSON arguments",
                    extra={"tool_name": definition.name, "alias": alias, "run_id": context.run_id, "task_id": context.task_id},
                )
                return _json_tool_output(
                    {
                        "tool_name": definition.name,
                        "tool_alias": alias,
                        "ok": False,
                        "error": "INVALID_TOOL_ARGUMENTS",
                        "observation": str(exc),
                        "truncated": False,
                    }
                )

            try:
                mapped_arguments = _map_sandbox_tool_arguments(definition.name, arguments, context)
            except SandboxPathError as exc:
                logger.info(
                    "SDK child-agent tool blocked sandbox path escape",
                    extra={"tool_name": definition.name, "alias": alias, "run_id": context.run_id, "task_id": context.task_id},
                )
                return _json_tool_output(
                    {
                        "tool_name": definition.name,
                        "tool_alias": alias,
                        "ok": False,
                        "error": "PATH_OUTSIDE_SANDBOX",
                        "observation": str(exc),
                        "truncated": False,
                    }
                )

            tool_call_id = f"sdk-tool:{uuid4().hex}"
            if _requires_dbx_connection_discovery(context, definition.name) and not _has_successful_dbx_connection_discovery(context):
                # Connection selection is a runtime invariant. The model may choose
                # which discovered connection to use, but it cannot bypass discovery.
                logger.warning(
                    "Blocked DBX MCP call before connection discovery",
                    extra={
                        "tool_name": definition.name,
                        "run_id": context.run_id,
                        "task_id": context.task_id,
                        "error_code": "DBX_CONNECTION_DISCOVERY_REQUIRED",
                    },
                )
                _emit_nested_mcp_event(
                    context,
                    event_type="subagent_tool_started",
                    tool_call_id=tool_call_id,
                    tool_name=definition.name,
                    status="blocked",
                    summary=f"{_child_agent_name(context)} 的 DBX 调用被拦截：请先发现连接。",
                    tool_input_keys=sorted(str(key) for key in mapped_arguments),
                )
                _record_runtime_mcp_call(
                    context,
                    call_id=tool_call_id,
                    tool_name=definition.name,
                    status="failed",
                    ok=False,
                )
                _emit_nested_mcp_event(
                    context,
                    event_type="subagent_tool_finished",
                    tool_call_id=tool_call_id,
                    tool_name=definition.name,
                    status="failed",
                    summary=f"{definition.name} 被拦截（DBX_CONNECTION_DISCOVERY_REQUIRED）。",
                    tool_input_keys=sorted(str(key) for key in mapped_arguments),
                )
                return _json_tool_output(
                    {
                        "tool_name": definition.name,
                        "tool_alias": alias,
                        "ok": False,
                        "error": "DBX_CONNECTION_DISCOVERY_REQUIRED",
                        "observation": "必须先调用 dbx_list_connections，再使用其他 DBX 工具。",
                        "truncated": False,
                    }
                )

            try:
                if definition.name.startswith("mcp."):
                    _emit_nested_mcp_event(
                        context,
                        event_type="subagent_tool_started",
                        tool_call_id=tool_call_id,
                        tool_name=definition.name,
                        status="running",
                        summary=f"{_child_agent_name(context)} 开始调用 {definition.name}。",
                        tool_input_keys=sorted(str(key) for key in mapped_arguments),
                    )
                raw_payload = definition.handler(self._session_for(context), **mapped_arguments)
            except Exception as exc:
                logger.exception(
                    "SDK child-agent delegated tool handler failed",
                    extra={"tool_name": definition.name, "alias": alias, "run_id": context.run_id, "task_id": context.task_id},
                )
                if definition.name.startswith("mcp."):
                    _record_runtime_mcp_call(
                        context,
                        call_id=tool_call_id,
                        tool_name=definition.name,
                        status="failed",
                        ok=False,
                    )
                    _emit_nested_mcp_event(
                        context,
                        event_type="subagent_tool_finished",
                        tool_call_id=tool_call_id,
                        tool_name=definition.name,
                        status="failed",
                        summary=f"{definition.name} 执行失败（{type(exc).__name__}）。",
                        tool_input_keys=sorted(str(key) for key in mapped_arguments),
                    )
                return _json_tool_output(
                    {
                        "tool_name": definition.name,
                        "tool_alias": alias,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "observation": f"Tool handler raised {type(exc).__name__}.",
                        "truncated": False,
                    }
                )

            payload = _payload_to_dict(raw_payload)
            observation, truncated = _compact_observation(payload, max_chars=self.max_observation_chars)
            output: dict[str, Any] = {
                "tool_name": definition.name,
                "tool_alias": alias,
                "ok": bool(payload.get("ok", True)),
                "observation": observation,
                "truncated": truncated,
            }
            if definition.name.startswith("mcp."):
                # Runtime evidence is generated here, after the handler returns.
                # Model-provided trace fields are never treated as execution proof.
                _record_runtime_mcp_call(
                    context,
                    call_id=tool_call_id,
                    tool_name=definition.name,
                    status="succeeded" if output["ok"] else "failed",
                    ok=output["ok"],
                )
            if payload.get("error"):
                output["error"] = payload.get("error")
            logger.info(
                "SDK child-agent delegated tool completed",
                extra={
                    "tool_name": definition.name,
                    "alias": alias,
                    "run_id": context.run_id,
                    "task_id": context.task_id,
                    "ok": output["ok"],
                    "truncated": truncated,
                },
            )
            if definition.name.startswith("mcp."):
                _emit_nested_mcp_event(
                    context,
                    event_type="subagent_tool_finished",
                    tool_call_id=tool_call_id,
                    tool_name=definition.name,
                    status="succeeded" if output["ok"] else "failed",
                    summary=f"{definition.name} 已执行" if output["ok"] else f"{definition.name} 返回错误。",
                    tool_input_keys=sorted(str(key) for key in mapped_arguments),
                )
            return _json_tool_output(output)

        needs_approval = bool(
            definition.requires_confirmation
            or str(getattr(definition.risk_level, "value", definition.risk_level)).strip().lower() == AgentToolRiskLevel.HIGH.value
        )
        return function_tool_cls(
            name=alias,
            description=f"{definition.description}\nRegistry tool name: {definition.name}",
            params_json_schema=dict(definition.input_schema),
            on_invoke_tool=invoke_tool,
            strict_json_schema=False,
            needs_approval=needs_approval,
        )

    def _session_for(self, context: AgentRuntimeContext) -> Any:
        if self.session_provider is None:
            return None
        return self.session_provider(context)


def _child_agent_name(context: AgentRuntimeContext) -> str:
    return str(getattr(context, "agent_name", None) or getattr(context, "child_agent_name", None) or "子 Agent")


def _requires_dbx_connection_discovery(context: AgentRuntimeContext, tool_name: str) -> bool:
    normalized_name = str(tool_name or "").strip().lower()
    capability_id = str(getattr(context, "capability_id", None) or "").strip()
    return (
        capability_id == DBX_READONLY_CAPABILITY_ID
        and normalized_name.startswith("mcp.dbx.")
        and normalized_name != DBX_CONNECTION_DISCOVERY_TOOL
    )


def _has_successful_dbx_connection_discovery(context: AgentRuntimeContext) -> bool:
    calls = context.metadata.get(SDK_AGENT_RUNTIME_MCP_CALLS_KEY, [])
    if not isinstance(calls, list):
        return False
    return any(
        isinstance(call, dict)
        and str(call.get("tool_name") or "").strip().lower() == DBX_CONNECTION_DISCOVERY_TOOL
        and bool(call.get("ok"))
        and str(call.get("status") or "").strip().lower() == "succeeded"
        for call in calls
    )


def _emit_nested_mcp_event(
    context: AgentRuntimeContext,
    *,
    event_type: str,
    tool_call_id: str,
    tool_name: str,
    status: str,
    summary: str,
    tool_input_keys: list[str],
) -> None:
    event_sink = getattr(context, "event_sink", None)
    if not callable(event_sink):
        return
    capability_id = getattr(context, "capability_id", None)
    agent_name = _child_agent_name(context)
    try:
        event_sink(
            {
                "event_type": event_type,
                "event_label": "子 Agent MCP 工具开始" if event_type == "subagent_tool_started" else "子 Agent MCP 工具完成",
                "session_id": context.session_id,
                "workflow_run_id": context.run_id,
                "agent_run_id": context.metadata.get("agent_run_id"),
                "delegation_id": context.metadata.get("delegation_id"),
                "step_index": None,
                "tool_name": tool_name,
                "capability": capability_id,
                "capability_kind": "agent",
                "executor_id": getattr(context, "executor_id", None) or "openai-sdk-agent",
                "tool_call_id": tool_call_id,
                "status": status,
                "summary": summary,
                "tool_input_keys": tool_input_keys,
                "parent_capability": capability_id,
                "parent_agent_name": agent_name,
            }
        )
    except Exception:  # UI/event delivery must not change MCP tool execution.
        logger.exception(
            "Failed to emit nested MCP tool event",
            extra={"tool_name": tool_name, "capability_id": capability_id, "run_id": context.run_id},
        )


def _record_runtime_mcp_call(
    context: AgentRuntimeContext,
    *,
    call_id: str,
    tool_name: str,
    status: str,
    ok: bool,
) -> None:
    """Record only runtime-owned MCP facts, never tool result contents."""

    calls = context.metadata.setdefault(SDK_AGENT_RUNTIME_MCP_CALLS_KEY, [])
    if not isinstance(calls, list):
        calls = []
        context.metadata[SDK_AGENT_RUNTIME_MCP_CALLS_KEY] = calls
    calls.append(
        {
            "call_id": call_id,
            "tool_name": tool_name,
            "status": status,
            "ok": bool(ok),
        }
    )


def _safe_tool_alias(tool_name: str) -> str:
    alias = re.sub(r"[^A-Za-z0-9_]", "_", tool_name).strip("_")[:64]
    return alias or "agent_tool"


def _parse_tool_arguments(arguments_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tool arguments must be a JSON object: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return payload


def _map_sandbox_tool_arguments(tool_name: str, arguments: dict[str, Any], context: AgentRuntimeContext) -> dict[str, Any]:
    workspace = _sandbox_workspace_from_context(context)
    if workspace is None or tool_name not in _SANDBOX_FILE_TOOLS:
        return dict(arguments)

    manager = SdkAgentSandboxManager(base_dir=workspace.root.parent, mode=workspace.mode)
    mapped = dict(arguments)
    if tool_name in _SANDBOX_READ_PATH_TOOLS:
        mapped["path"] = str(manager.resolve_path(workspace, _required_argument(arguments, "path"), access="read"))
        return mapped
    if tool_name in _SANDBOX_SINGLE_PATH_WRITE_TOOLS:
        mapped["path"] = str(
            manager.resolve_path(
                workspace,
                _required_argument(arguments, "path"),
                access=_SANDBOX_SINGLE_PATH_WRITE_TOOLS[tool_name],
            )
        )
        return mapped
    if tool_name == FILESYSTEM_COPY_FILE_TOOL:
        mapped["src"] = str(manager.resolve_path(workspace, _required_argument(arguments, "src"), access="read"))
        mapped["dst"] = str(manager.resolve_path(workspace, _required_argument(mapped, "dst"), access="write"))
        return mapped
    if tool_name == FILESYSTEM_MOVE_FILE_TOOL:
        mapped["src"] = str(manager.resolve_path(workspace, _required_argument(arguments, "src"), access="move"))
        mapped["dst"] = str(manager.resolve_path(workspace, _required_argument(arguments, "dst"), access="write"))
        return mapped
    return mapped


def _sandbox_workspace_from_context(context: AgentRuntimeContext) -> SdkAgentSandboxWorkspace | None:
    payload = context.metadata.get(SDK_AGENT_SANDBOX_METADATA_KEY)
    if isinstance(payload, SdkAgentSandboxWorkspace):
        return payload
    if not isinstance(payload, dict) or not payload.get("root"):
        return None
    try:
        return SdkAgentSandboxWorkspace.from_metadata(payload)
    except Exception:
        return None


def _required_argument(arguments: dict[str, Any], name: str) -> Any:
    value = arguments.get(name)
    if value is None:
        raise SandboxPathError(f"Missing required sandbox path argument: {name}.")
    return value


def _payload_to_dict(raw_payload: Any) -> dict[str, Any]:
    if isinstance(raw_payload, dict):
        return dict(raw_payload)
    if is_dataclass(raw_payload):
        return asdict(raw_payload)
    model_dump = getattr(raw_payload, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(mode="json"))
    return {"ok": True, "result": raw_payload}


def _compact_observation(payload: dict[str, Any], *, max_chars: int) -> tuple[str, bool]:
    limit = max(200, int(max_chars or DEFAULT_SDK_TOOL_OBSERVATION_CHARS))
    text = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)
    if len(text) <= limit:
        return text, False
    return f"{text[:limit].rstrip()}...", True


def _json_tool_output(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)


__all__ = ["DEFAULT_SDK_TOOL_OBSERVATION_CHARS", "SdkAgentToolGateway"]
