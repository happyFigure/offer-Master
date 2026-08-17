from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.mcp_gateway.tool_policy import MCPToolPolicy


class AgentToolRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class AgentToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    handler: Callable[..., Any] | None = None
    risk_level: AgentToolRiskLevel = AgentToolRiskLevel.LOW
    requires_confirmation: bool = False
    allowed_source_types: frozenset[str] = field(default_factory=frozenset)
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Agent tool name is required")
        if not self.description.strip():
            raise ValueError(f"Agent tool description is required: {self.name}")
        object.__setattr__(self, "allowed_source_types", frozenset(self.allowed_source_types))


class AgentToolRegistry:
    def __init__(self, definitions: Iterable[AgentToolDefinition] | None = None) -> None:
        self._definitions: dict[str, AgentToolDefinition] = {}
        for definition in definitions or ():
            self.register(definition)

    def register(self, definition: AgentToolDefinition) -> AgentToolDefinition:
        if definition.name in self._definitions:
            raise ValueError(f"Agent tool already registered: {definition.name}")
        self._definitions[definition.name] = definition
        return definition

    def register_many(self, definitions: Iterable[AgentToolDefinition]) -> None:
        for definition in definitions:
            self.register(definition)

    def get(self, name: str) -> AgentToolDefinition | None:
        definition = self._definitions.get(name)
        if definition is None or not definition.enabled:
            return None
        return definition

    def list_definitions(self) -> list[AgentToolDefinition]:
        return sorted(
            (definition for definition in self._definitions.values() if definition.enabled),
            key=lambda definition: definition.name,
        )

    def registered_tool_names(self) -> list[str]:
        return [definition.name for definition in self.list_definitions()]


def create_default_agent_tool_registry(*, content_source_client: Any | None = None) -> AgentToolRegistry:
    registry = AgentToolRegistry()
    registry.register_many(_memory_tool_definitions())
    registry.register_many(create_content_source_agent_tool_definitions(content_source_client))
    return registry


def create_content_source_agent_tool_definitions(client: Any | None = None) -> list[AgentToolDefinition]:
    from app.mcp_gateway.content_source_client import ContentSourceMCPClient

    content_client = client or ContentSourceMCPClient()
    return [
        AgentToolDefinition(
            name="weixin-articles-mcp.read_article",
            description="Read one public WeChat official-account article URL and return extracted text/media blocks.",
            input_schema={
                "type": "object",
                "required": ["url"],
                "properties": {"url": {"type": "string", "description": "Public mp.weixin.qq.com article URL."}},
            },
            output_schema={"type": "object", "required": ["tool_name", "ok"]},
            handler=lambda _session, **arguments: content_client.read_weixin_article(url=str(arguments.get("url") or "")),
            allowed_source_types=frozenset({"agent_chat", "wechat_article", "wechat_account"}),
        ),
        AgentToolDefinition(
            name="xiaohongshu-mcp.search_feeds",
            description="Search Xiaohongshu feeds for recruiting-related notes by keyword through MCP Gateway.",
            input_schema={
                "type": "object",
                "required": ["keyword"],
                "properties": {
                    "keyword": {"type": "string"},
                    "filters": {"type": ["object", "null"], "additionalProperties": True},
                },
            },
            output_schema={"type": "object", "required": ["tool_name", "ok"]},
            handler=lambda _session, **arguments: content_client.search_xiaohongshu_feeds(
                keyword=str(arguments.get("keyword") or ""),
                filters=arguments.get("filters") if isinstance(arguments.get("filters"), dict) else None,
            ),
            allowed_source_types=frozenset({"agent_chat", "xiaohongshu_note", "mcp_visible_page"}),
        ),
        AgentToolDefinition(
            name="xiaohongshu-mcp.get_feed_detail",
            description="Read one Xiaohongshu feed detail through MCP Gateway using feed_id and xsec_token.",
            input_schema={
                "type": "object",
                "required": ["feed_id", "xsec_token"],
                "properties": {
                    "feed_id": {"type": "string"},
                    "xsec_token": {"type": "string"},
                    "include_comments": {"type": "boolean"},
                    "comment_limit": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                "additionalProperties": True,
            },
            output_schema={"type": "object", "required": ["tool_name", "ok"]},
            handler=lambda _session, **arguments: content_client.get_xiaohongshu_feed_detail(**arguments),
            allowed_source_types=frozenset({"agent_chat", "xiaohongshu_note", "mcp_visible_page"}),
        ),
    ]


def create_mcp_agent_tool_definitions(client: Any, *, allowed_tool_names: Iterable[str]) -> list[AgentToolDefinition]:
    policy = MCPToolPolicy.from_allowlist(allowed_tool_names)
    definitions: list[AgentToolDefinition] = []
    for tool_name in policy.allowed_tool_names():
        definitions.append(
            AgentToolDefinition(
                name=f"mcp.{tool_name}",
                description=f"Call MCP Gateway tool: {tool_name}.",
                input_schema={"type": "object", "additionalProperties": True},
                output_schema={"type": "object", "required": ["tool_name", "ok"]},
                handler=_mcp_handler(client, tool_name),
                risk_level=_mcp_risk_level(policy, tool_name),
                requires_confirmation=policy.requires_confirmation(tool_name),
                allowed_source_types=frozenset({"agent_chat", "mcp_visible_page", "application"}),
            )
        )
    return definitions


def _memory_tool_definitions() -> list[AgentToolDefinition]:
    from app.agent_runtime.memory.memory_tools import memory_get, memory_search, sessions_history, sessions_search

    return [
        AgentToolDefinition(
            name="sessions_search",
            description="Search prior agent session transcript messages and context summaries.",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
            },
            output_schema={"type": "object", "required": ["corpus", "query", "items"]},
            handler=sessions_search,
            allowed_source_types=frozenset({"agent_chat", "agent_session", "history_recall"}),
        ),
        AgentToolDefinition(
            name="sessions_history",
            description="Read a bounded message window around a prior session message.",
            input_schema={
                "type": "object",
                "required": ["session_key"],
                "properties": {
                    "session_key": {"type": "string"},
                    "around_message_id": {"type": ["string", "null"]},
                    "window_before": {"type": "integer", "minimum": 0, "maximum": 50},
                    "window_after": {"type": "integer", "minimum": 0, "maximum": 50},
                },
            },
            output_schema={"type": "object", "required": ["session_id", "messages"]},
            handler=sessions_history,
            allowed_source_types=frozenset({"agent_chat", "agent_session", "history_recall"}),
        ),
        AgentToolDefinition(
            name="memory_search",
            description="Search long-term semantic memories and skill records only.",
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "corpus": {"type": ["string", "null"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
            },
            output_schema={"type": "object", "required": ["corpus", "query", "items"]},
            handler=memory_search,
            allowed_source_types=frozenset({"agent_chat", "long_term_memory", "skill_recall"}),
        ),
        AgentToolDefinition(
            name="memory_get",
            description="Read one precise long-term memory or skill record by id.",
            input_schema={
                "type": "object",
                "required": ["memory_id"],
                "properties": {"memory_id": {"type": "string"}},
            },
            output_schema={"type": "object", "required": ["memory_id", "found"]},
            handler=memory_get,
            allowed_source_types=frozenset({"agent_chat", "long_term_memory", "skill_recall"}),
        ),
    ]


def _mcp_handler(client: Any, tool_name: str) -> Callable[..., Any]:
    def handler(_session: Any, **arguments: Any) -> Any:
        return client.call_tool(tool_name=tool_name, arguments=arguments)

    return handler


def _mcp_risk_level(policy: MCPToolPolicy, tool_name: str) -> AgentToolRiskLevel:
    if policy.requires_confirmation(tool_name):
        return AgentToolRiskLevel.HIGH
    if tool_name in {"open_page", "read_page"}:
        return AgentToolRiskLevel.LOW
    return AgentToolRiskLevel.MEDIUM
