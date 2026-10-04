from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY
from app.agent_runtime.sdk_agents.sandbox import SDK_AGENT_SANDBOX_MODE_TEMP_COPY
from app.agent_runtime.tool_registry import (
    ARTIFACT_EXPORT_TOOL,
    EXTERNAL_WEB_SEARCH_TOOL,
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
    SKILL_LIST_ACTIONS_TOOL,
    SKILL_LIST_TOOL,
    SKILL_READ_TOOL,
    AgentToolRegistry,
)


SDK_AGENT_FILE_READ_TOOLS = (
    FILESYSTEM_LIST_DIR_TOOL,
    FILESYSTEM_PATH_EXISTS_TOOL,
    FILESYSTEM_PATH_STAT_TOOL,
    FILESYSTEM_READ_FILE_TOOL,
)
SDK_AGENT_SANDBOX_MUTATION_TOOLS = (
    FILESYSTEM_WRITE_TEXT_TOOL,
    FILESYSTEM_REPLACE_TEXT_TOOL,
    FILESYSTEM_COPY_FILE_TOOL,
    FILESYSTEM_MOVE_FILE_TOOL,
    FILESYSTEM_DELETE_PATH_TOOL,
    FILESYSTEM_MAKE_DIR_TOOL,
)
SDK_AGENT_RUNTIME_RETAINED_TOOLS = (ARTIFACT_EXPORT_TOOL,)
SDK_AGENT_SKILL_DISCOVERY_TOOLS = (SKILL_LIST_TOOL, SKILL_LIST_ACTIONS_TOOL, SKILL_READ_TOOL)
SDK_AGENT_CHROME_MCP_PREFIXES = ("mcp.chrome.", "mcp.google_chrome.", "mcp.google-chrome.")
SDK_AGENT_DBX_MCP_PREFIXES = ("mcp.dbx.", "mcp.dropbox.")
SDK_AGENT_QQ_MAIL_MCP_PREFIXES = ("mcp.qq_mail.", "mcp.qq-mail.", "mcp.qqmail.")
SDK_AGENT_CHILD_MCP_PREFIXES = (*SDK_AGENT_CHROME_MCP_PREFIXES, *SDK_AGENT_DBX_MCP_PREFIXES, *SDK_AGENT_QQ_MAIL_MCP_PREFIXES)
GOOGLE_CHROME_AGENT_CAPABILITY = "agent.google_chrome"
DBX_READONLY_AGENT_CAPABILITY = "agent.dbx_readonly"
QQ_MAIL_READONLY_AGENT_CAPABILITY = "agent.qq_mail_readonly"


@dataclass(frozen=True)
class SdkAgentDelegationPolicy:
    internal_tool_names: tuple[str, ...]
    exposed_capability_names: tuple[str, ...]
    runtime_retained_tool_names: tuple[str, ...] = SDK_AGENT_RUNTIME_RETAINED_TOOLS
    reasons_by_tool: dict[str, str] = field(default_factory=dict)
    tools_by_capability: dict[str, tuple[str, ...]] = field(default_factory=dict)
    subagent_names_by_capability: dict[str, str] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "internal_tools": list(self.internal_tool_names),
            "exposed_capabilities": list(self.exposed_capability_names),
            "runtime_retained_tools": list(self.runtime_retained_tool_names),
            "reasons_by_tool": dict(self.reasons_by_tool),
            "tools_by_capability": {key: list(value) for key, value in self.tools_by_capability.items()},
            "subagent_names_by_capability": dict(self.subagent_names_by_capability),
        }


def build_sdk_agent_delegation_policy(
    settings: Any,
    *,
    tool_registry: AgentToolRegistry | None = None,
) -> SdkAgentDelegationPolicy:
    requested = _requested_sdk_internal_tools(settings, tool_registry=tool_registry)
    retained = tuple(_registered_or_requested(SDK_AGENT_RUNTIME_RETAINED_TOOLS, tool_registry=tool_registry))
    internal_tools = tuple(
        name
        for name in _registered_or_requested(requested, tool_registry=tool_registry)
        if name not in set(retained)
    )
    exposed_capabilities = tuple(
        name
        for name in internal_tools
        if not _is_mcp_tool(name) and _can_expose_as_runtime_capability(name, tool_registry=tool_registry)
    )
    if _should_delegate_filesystem_skill(settings, internal_tools=internal_tools):
        exposed_capabilities = tuple(dict.fromkeys([*exposed_capabilities, FILESYSTEM_SKILL_CAPABILITY]))
    reasons = {name: "runtime_confirmation_boundary" for name in retained}
    if FILESYSTEM_SKILL_CAPABILITY in exposed_capabilities:
        reasons[FILESYSTEM_SKILL_CAPABILITY] = "child_agent_skill_orchestration"

    tools_by_capability: dict[str, tuple[str, ...]] = {}
    subagent_names_by_capability: dict[str, str] = {}
    chrome_tools = tuple(name for name in internal_tools if _has_prefix(name, SDK_AGENT_CHROME_MCP_PREFIXES))
    dbx_tools = tuple(name for name in internal_tools if _has_prefix(name, SDK_AGENT_DBX_MCP_PREFIXES))
    qq_mail_tools = tuple(
        name
        for name in internal_tools
        if _has_prefix(name, SDK_AGENT_QQ_MAIL_MCP_PREFIXES)
        and str(name).rsplit(".", 1)[-1] in {"list_messages", "get_message"}
    )
    if chrome_tools:
        tools_by_capability[GOOGLE_CHROME_AGENT_CAPABILITY] = chrome_tools
        subagent_names_by_capability[GOOGLE_CHROME_AGENT_CAPABILITY] = "GoogleChromeAgent"
        exposed_capabilities = tuple(dict.fromkeys([*exposed_capabilities, GOOGLE_CHROME_AGENT_CAPABILITY]))
    if dbx_tools:
        tools_by_capability[DBX_READONLY_AGENT_CAPABILITY] = dbx_tools
        subagent_names_by_capability[DBX_READONLY_AGENT_CAPABILITY] = "DbxReadOnlyAgent"
        exposed_capabilities = tuple(dict.fromkeys([*exposed_capabilities, DBX_READONLY_AGENT_CAPABILITY]))
    if qq_mail_tools:
        mail_tools = tuple(
            sorted(
                qq_mail_tools,
                key=lambda name: {"list_messages": 0, "get_message": 1}.get(str(name).rsplit(".", 1)[-1], 99),
            )
        )
        skill_tools = tuple(name for name in internal_tools if name in SDK_AGENT_SKILL_DISCOVERY_TOOLS)
        tools_by_capability[QQ_MAIL_READONLY_AGENT_CAPABILITY] = (*skill_tools, *mail_tools)
        subagent_names_by_capability[QQ_MAIL_READONLY_AGENT_CAPABILITY] = "QqMailReadOnlyAgent"
        exposed_capabilities = tuple(dict.fromkeys([*exposed_capabilities, QQ_MAIL_READONLY_AGENT_CAPABILITY]))
    for name in internal_tools:
        if name in SDK_AGENT_SKILL_DISCOVERY_TOOLS:
            reasons[name] = "child_agent_skill_discovery"
        if name in SDK_AGENT_FILE_READ_TOOLS:
            # The child agent uses concrete filesystem tools internally, while
            # the runtime exposes only the coarse filesystem Skill as the route.
            reasons[name] = "child_internal_filesystem_tool"
        if _is_enabled_mcp_tool(name, settings=settings):
            reasons[name] = "child_agent_mcp_tool"
        if name not in exposed_capabilities and _is_high_risk_tool(name, tool_registry=tool_registry):
            # High-risk sandbox tools may be used inside a child run, but they are not advertised as top-level routes.
            reasons[name] = "child_internal_sandbox_only"
    return SdkAgentDelegationPolicy(
        internal_tool_names=internal_tools,
        exposed_capability_names=exposed_capabilities,
        runtime_retained_tool_names=retained,
        reasons_by_tool=reasons,
        tools_by_capability=tools_by_capability,
        subagent_names_by_capability=subagent_names_by_capability,
    )


def _requested_sdk_internal_tools(settings: Any, *, tool_registry: AgentToolRegistry | None = None) -> tuple[str, ...]:
    tool_names: list[str] = []
    if bool(getattr(settings, "sdk_agent_enable_web_research", False)):
        tool_names.append(EXTERNAL_WEB_SEARCH_TOOL)
    if bool(getattr(settings, "sdk_agent_enable_file_analysis", False)):
        tool_names.extend(SDK_AGENT_SKILL_DISCOVERY_TOOLS)
        tool_names.extend(SDK_AGENT_FILE_READ_TOOLS)
    if (
        bool(getattr(settings, "sdk_agent_enable_mutation_tools", False))
        and str(getattr(settings, "sdk_agent_sandbox_mode", "none")) == SDK_AGENT_SANDBOX_MODE_TEMP_COPY
    ):
        tool_names.extend(SDK_AGENT_SANDBOX_MUTATION_TOOLS)
    if bool(getattr(settings, "sdk_agent_enable_chrome_mcp", False)):
        tool_names.extend(_registered_mcp_tools(tool_registry, prefixes=SDK_AGENT_CHROME_MCP_PREFIXES))
    if bool(getattr(settings, "sdk_agent_enable_dbx_mcp", False)):
        tool_names.extend(_registered_mcp_tools(tool_registry, prefixes=SDK_AGENT_DBX_MCP_PREFIXES))
    if bool(getattr(settings, "sdk_agent_enable_qq_mail_mcp", False)):
        tool_names.extend(SDK_AGENT_SKILL_DISCOVERY_TOOLS)
        tool_names.extend(_registered_mcp_tools(tool_registry, prefixes=SDK_AGENT_QQ_MAIL_MCP_PREFIXES))
    return tuple(dict.fromkeys(tool_names))


def _registered_or_requested(tool_names: tuple[str, ...], *, tool_registry: AgentToolRegistry | None) -> tuple[str, ...]:
    if tool_registry is None:
        return tool_names
    return tuple(name for name in tool_names if tool_registry.get(name) is not None)


def _can_expose_as_runtime_capability(name: str, *, tool_registry: AgentToolRegistry | None) -> bool:
    if _is_mcp_tool(name):
        return False
    if name in SDK_AGENT_FILE_READ_TOOLS:
        return False
    if name in SDK_AGENT_SKILL_DISCOVERY_TOOLS:
        return False
    if tool_registry is None:
        return False
    definition = tool_registry.get(name)
    if definition is None or definition.handler is None or definition.requires_confirmation:
        return False
    return str(getattr(definition.risk_level, "value", definition.risk_level)).strip().lower() == "low"


def _should_delegate_filesystem_skill(settings: Any, *, internal_tools: tuple[str, ...]) -> bool:
    if not bool(getattr(settings, "sdk_agent_enable_file_analysis", False)):
        return False
    return any(name in internal_tools for name in SDK_AGENT_FILE_READ_TOOLS)


def _registered_mcp_tools(
    tool_registry: AgentToolRegistry | None,
    *,
    prefixes: tuple[str, ...],
) -> tuple[str, ...]:
    if tool_registry is None:
        return ()
    definitions = getattr(tool_registry, "list_definitions", lambda: [])()
    names = [str(getattr(definition, "name", "") or "").strip() for definition in definitions]
    selected = [name for name in names if name and any(name.startswith(prefix) for prefix in prefixes)]
    if any(prefix in SDK_AGENT_QQ_MAIL_MCP_PREFIXES for prefix in prefixes):
        selected = [name for name in selected if name.rsplit(".", 1)[-1] in {"list_messages", "get_message"}]
    return tuple(selected)


def _is_mcp_tool(name: str) -> bool:
    return str(name or "").startswith("mcp.")


def _has_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(str(name or "").startswith(prefix) for prefix in prefixes)


def _is_enabled_mcp_tool(name: str, *, settings: Any) -> bool:
    if bool(getattr(settings, "sdk_agent_enable_chrome_mcp", False)) and any(name.startswith(prefix) for prefix in SDK_AGENT_CHROME_MCP_PREFIXES):
        return True
    if bool(getattr(settings, "sdk_agent_enable_dbx_mcp", False)) and any(name.startswith(prefix) for prefix in SDK_AGENT_DBX_MCP_PREFIXES):
        return True
    if bool(getattr(settings, "sdk_agent_enable_qq_mail_mcp", False)) and any(name.startswith(prefix) for prefix in SDK_AGENT_QQ_MAIL_MCP_PREFIXES):
        return True
    return False


def _is_high_risk_tool(name: str, *, tool_registry: AgentToolRegistry | None) -> bool:
    if tool_registry is None:
        return name in SDK_AGENT_SANDBOX_MUTATION_TOOLS or name in SDK_AGENT_RUNTIME_RETAINED_TOOLS
    definition = tool_registry.get(name)
    if definition is None:
        return False
    return definition.requires_confirmation or str(getattr(definition.risk_level, "value", definition.risk_level)).strip().lower() == "high"


__all__ = [
    "SDK_AGENT_FILE_READ_TOOLS",
    "SDK_AGENT_CHILD_MCP_PREFIXES",
    "SDK_AGENT_RUNTIME_RETAINED_TOOLS",
    "SDK_AGENT_SANDBOX_MUTATION_TOOLS",
    "SDK_AGENT_SKILL_DISCOVERY_TOOLS",
    "GOOGLE_CHROME_AGENT_CAPABILITY",
    "DBX_READONLY_AGENT_CAPABILITY",
    "QQ_MAIL_READONLY_AGENT_CAPABILITY",
    "SdkAgentDelegationPolicy",
    "build_sdk_agent_delegation_policy",
]
