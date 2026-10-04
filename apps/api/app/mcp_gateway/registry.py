from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
import time
from typing import Any, Iterable
from uuid import uuid4

from app.mcp_gateway.client import MCPToolCallResult
from app.mcp_gateway.dbx_readonly import filter_mcp_tool_names, validate_dbx_tool_arguments
from app.mcp_gateway.qq_mail_readonly import validate_qq_mail_tool_arguments
from app.mcp_gateway.registry_config import MCPServerConfig
from app.mcp_gateway.stdio_client import MCPStdioServerSpec, StdioMCPGatewayClient


logger = logging.getLogger(__name__)
_DISCOVERY_CACHE: dict[str, tuple[float, str, str, list[dict[str, Any]]]] = {}
_DISCOVERY_CACHE_TTL_SECONDS = 60.0


@dataclass
class MCPServerRuntime:
    config: MCPServerConfig
    status: str = "configured"
    detail: str = "Configured; tool discovery has not run."
    discovered_tools: dict[str, dict[str, Any]] = field(default_factory=dict)


class MCPRegistryClient:
    """One runtime boundary for configured MCP servers.

    The registry owns server selection and allow/deny filtering. The Agent tool
    registry only receives the resulting definitions, so child agents cannot
    start an arbitrary MCP process or bypass runtime policy.
    """

    def __init__(
        self,
        *,
        servers: Iterable[MCPServerConfig],
        allowed_tool_names: Iterable[str] = (),
        transport: Any | None = None,
    ) -> None:
        self._runtimes = {
            config.id: MCPServerRuntime(config=config)
            for config in servers
            if config.enabled
        }
        specs = tuple(
            MCPStdioServerSpec(
                tool_prefix=runtime.config.tool_prefix,
                command=runtime.config.command,
                args=runtime.config.args,
                env=runtime.config.env,
                cwd=runtime.config.cwd,
                timeout_seconds=runtime.config.timeout_seconds,
            )
            for runtime in self._runtimes.values()
            if runtime.config.transport == "stdio"
        )
        self._gateway = StdioMCPGatewayClient(servers=specs, transport=transport)
        self._settings_allowlist = set(str(name).strip() for name in allowed_tool_names if str(name).strip())
        self._discovery_attempted = False
        # Real stdio startup is cached between chat turns; injected test
        # transports stay isolated and never share discovery state.
        self._cache_discovery = transport is None

    def call_tool(self, *, tool_name: str, arguments: dict[str, Any]) -> MCPToolCallResult:
        dispatch_id = f"mcp-dispatch:{uuid4().hex}"
        log_context = {"dispatch_id": dispatch_id, "mcp_tool_name": tool_name}
        logger.info("MCP registry dispatch started", extra=log_context)

        def finish(result: MCPToolCallResult, *, server_id: str | None = None) -> MCPToolCallResult:
            logger.info(
                "MCP registry dispatch finished",
                extra={
                    **log_context,
                    "mcp_server_id": server_id,
                    "ok": bool(result.ok),
                    "error_code": _safe_error_code(result.error),
                },
            )
            return result

        server, local_name = self._resolve(tool_name)
        if server is None or local_name is None:
            return finish(
                MCPToolCallResult(
                    tool_name=tool_name,
                    ok=False,
                    error="MCP_TOOL_NOT_REGISTERED",
                    result={"message": f"Tool is not registered in MCP runtime registry: {tool_name}"},
                )
            )
        normalized_tool_name = f"{server.config.tool_prefix}.{local_name}"
        if self._discovery_attempted and normalized_tool_name not in server.discovered_tools:
            return finish(
                MCPToolCallResult(
                    tool_name=tool_name,
                    ok=False,
                    error="MCP_TOOL_NOT_DISCOVERED",
                    result={"message": f"MCP server did not advertise this tool: {tool_name}"},
                ),
                server_id=server.config.id,
            )
        if not self._is_allowed(server.config, local_name, tool_name):
            return finish(
                MCPToolCallResult(
                    tool_name=tool_name,
                    ok=False,
                    error="MCP_TOOL_POLICY_DENIED",
                    result={"message": f"Tool is denied by MCP registry policy: {tool_name}"},
                ),
                server_id=server.config.id,
            )
        policy_error = validate_dbx_tool_arguments(tool_name, arguments)
        if policy_error is None:
            policy_error = validate_qq_mail_tool_arguments(tool_name, arguments)
        if policy_error is not None:
            logger.warning(
                "MCP registry blocked DBX call",
                extra={
                    "mcp_server_id": server.config.id,
                    "mcp_tool_name": tool_name,
                    "policy_error": policy_error.error,
                    "dispatch_id": dispatch_id,
                },
            )
            return finish(policy_error, server_id=server.config.id)
        try:
            result = self._gateway.call_tool(tool_name=tool_name, arguments=dict(arguments or {}))
        except Exception as exc:  # Keep registry callers on the standard result contract.
            logger.exception(
                "MCP registry dispatch raised",
                extra={
                    "mcp_server_id": server.config.id,
                    "mcp_tool_name": tool_name,
                    "dispatch_id": dispatch_id,
                    "error_type": type(exc).__name__,
                },
            )
            return finish(
                MCPToolCallResult(
                    tool_name=tool_name,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    result={"message": "MCP registry dispatch failed."},
                ),
                server_id=server.config.id,
            )
        return finish(result, server_id=server.config.id)

    def configured_tool_names(self) -> list[str]:
        names: list[str] = []
        for runtime in self._runtimes.values():
            for local_name in runtime.config.allow_tools:
                full_name = f"{runtime.config.tool_prefix}.{local_name}"
                if self._is_allowed(runtime.config, local_name, full_name) and full_name not in names:
                    names.append(full_name)
        return filter_mcp_tool_names(names)

    def registered_tool_names(self) -> list[str]:
        """Return tools with discovered schemas, or declared tools before discovery."""

        discovered = [name for runtime in self._runtimes.values() for name in runtime.discovered_tools]
        if self._discovery_attempted:
            return discovered
        return self.configured_tool_names()

    def discover_tools(self) -> list[dict[str, Any]]:
        """Ask enabled MCP servers for schemas and retain only allowed tools."""

        self._discovery_attempted = True
        discovered: list[dict[str, Any]] = []
        discovery_id = f"mcp-discovery:{uuid4().hex}"
        logger.info(
            "MCP registry discovery started",
            extra={"discovery_id": discovery_id, "server_count": len(self._runtimes)},
        )
        for runtime in self._runtimes.values():
            cache_key = _discovery_cache_key(runtime.config)
            cached = _DISCOVERY_CACHE.get(cache_key) if self._cache_discovery else None
            if cached is not None and time.monotonic() - cached[0] < _DISCOVERY_CACHE_TTL_SECONDS:
                _, runtime.status, runtime.detail, cached_tools = cached
                for tool in cached_tools:
                    runtime.discovered_tools[str(tool.get("name"))] = dict(tool)
                    discovered.append(dict(tool))
                logger.info(
                    "MCP registry discovery finished from cache",
                    extra={
                        "discovery_id": discovery_id,
                        "mcp_server_id": runtime.config.id,
                        "status": runtime.status,
                        "allowed_tool_count": len(runtime.discovered_tools),
                    },
                )
                continue
            try:
                tools = self._gateway.list_tools(prefix=runtime.config.tool_prefix)
            except Exception as exc:  # pragma: no cover - transport-specific.
                runtime.status = "unavailable"
                runtime.detail = f"Tool discovery failed: {type(exc).__name__}"
                logger.warning(
                    "MCP registry discovery failed",
                    extra={
                        "mcp_server_id": runtime.config.id,
                        "discovery_id": discovery_id,
                        "error_type": type(exc).__name__,
                    },
                )
                if self._cache_discovery:
                    _DISCOVERY_CACHE[cache_key] = (time.monotonic(), runtime.status, runtime.detail, [])
                continue
            for tool in tools:
                name = str(tool.get("name") or "").strip()
                local_name = name.split(".", 1)[-1] if "." in name else name
                if not name or not self._is_allowed(runtime.config, local_name, name):
                    continue
                runtime.discovered_tools[name] = dict(tool)
                discovered.append(dict(tool))
            runtime.status = "ready" if runtime.discovered_tools else "unavailable"
            runtime.detail = (
                f"Discovered {len(runtime.discovered_tools)} allowed tool(s)."
                if runtime.discovered_tools
                else "MCP server returned no allowed tools or could not be reached."
            )
            logger.info(
                "MCP registry discovery finished",
                extra={
                    "discovery_id": discovery_id,
                    "mcp_server_id": runtime.config.id,
                    "status": runtime.status,
                    "allowed_tool_count": len(runtime.discovered_tools),
                },
            )
            if self._cache_discovery:
                _DISCOVERY_CACHE[cache_key] = (
                    time.monotonic(),
                    runtime.status,
                    runtime.detail,
                    [dict(tool) for tool in runtime.discovered_tools.values()],
                )
        logger.info(
            "MCP registry discovery completed",
            extra={"discovery_id": discovery_id, "discovered_tool_count": len(discovered)},
        )
        return discovered

    def tool_metadata(self, tool_name: str) -> dict[str, Any] | None:
        for runtime in self._runtimes.values():
            metadata = runtime.discovered_tools.get(tool_name)
            if metadata is not None:
                return dict(metadata)
        return None

    def statuses(self) -> list[dict[str, Any]]:
        return [
            {
                "id": runtime.config.id,
                "tool_prefix": runtime.config.tool_prefix,
                "status": runtime.status,
                "detail": runtime.detail,
                "configured_tools": self.configured_tool_names_for(runtime.config.id),
                "discovered_tools": sorted(runtime.discovered_tools),
                "policy": runtime.config.policy,
            }
            for runtime in self._runtimes.values()
        ]

    def configured_tool_names_for(self, server_id: str) -> list[str]:
        runtime = self._runtimes.get(server_id)
        if runtime is None:
            return []
        return [
            f"{runtime.config.tool_prefix}.{local_name}"
            for local_name in runtime.config.allow_tools
            if self._is_allowed(runtime.config, local_name, f"{runtime.config.tool_prefix}.{local_name}")
        ]

    def _resolve(self, tool_name: str) -> tuple[MCPServerRuntime | None, str | None]:
        normalized = str(tool_name or "").strip()
        if "." not in normalized:
            return None, None
        prefix, local_name = normalized.split(".", 1)
        runtime = next((item for item in self._runtimes.values() if item.config.tool_prefix == prefix.lower()), None)
        return runtime, local_name or None

    def _is_allowed(self, config: MCPServerConfig, local_name: str, full_name: str) -> bool:
        if config.allow_tools and local_name not in config.allow_tools:
            return False
        if local_name in config.deny_tools or full_name in config.deny_tools:
            return False
        if self._settings_allowlist and full_name not in self._settings_allowlist:
            return False
        if config.policy == "dbx_read_only" and full_name.lower() not in {
            "dbx.dbx_list_connections",
            "dbx.dbx_list_databases",
            "dbx.dbx_list_tables",
            "dbx.dbx_get_schema_context",
            "dbx.dbx_describe_table",
            "dbx.dbx_execute_query",
        }:
            return False
        if config.policy == "qq_mail_read_only" and full_name.lower() not in {
            "qq_mail.list_messages",
            "qq_mail.get_message",
        }:
            return False
        return True


def _discovery_cache_key(config: MCPServerConfig) -> str:
    payload = "\x1f".join(
        [config.id, config.tool_prefix, config.command, *config.args, config.cwd or "", config.policy]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_error_code(value: Any) -> str | None:
    """Keep logs useful without copying exception details or tool payloads."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.split(":", 1)[0][:80]


__all__ = ["MCPRegistryClient", "MCPServerRuntime"]
