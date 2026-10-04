from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.mcp_gateway.client import HttpMCPGatewayClient, MCPGatewayClientProtocol
from app.mcp_gateway.registry import MCPRegistryClient
from app.mcp_gateway.registry_config import load_mcp_server_configs


def create_configured_mcp_client(settings: Any, *, discover: bool = False) -> MCPGatewayClientProtocol | None:
    """Build the MCP client used by runtime-registered MCP tools.

    OfferMaster originally only understood its own REST-shaped MCP Gateway.
    Chrome DevTools MCP and DBX MCP are ordinary stdio MCP servers, so when no
    REST gateway is configured we expose a small stdio bridge instead.
    """

    if bool(getattr(settings, "mcp_enabled", False)) and getattr(settings, "mcp_server_url", None):
        return HttpMCPGatewayClient(server_url=str(settings.mcp_server_url))

    servers = _load_registry_servers(settings)
    if not servers:
        return None
    # The declaration file is the source of truth for stdio MCP tools. The
    # legacy HTTP gateway keeps using JOBPILOT_MCP_TOOL_ALLOWLIST; applying
    # that generic list here would silently hide Chrome tools that were
    # explicitly allowed in config/mcp_servers.json.
    client = MCPRegistryClient(servers=servers)
    if discover and hasattr(client, "discover_tools"):
        client.discover_tools()
    return client


def configured_mcp_tool_names(client: Any, *, fallback: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Return declared names for registration; discovery remains an execution guard.

    A failed MCP discovery must not remove the child Agent capability from the
    runtime registry. The declaration is the permission boundary, while the
    registry still rejects actual calls for tools it did not discover.
    """

    declared_names = getattr(client, "configured_tool_names", None)
    if callable(declared_names):
        return list(declared_names())
    names = getattr(client, "registered_tool_names", None)
    if callable(names):
        return list(names())
    return [str(name).strip() for name in fallback if str(name).strip()]


def configured_mcp_transport_label(settings: Any, *, integration_id: str | None = None) -> str | None:
    if bool(getattr(settings, "mcp_enabled", False)) and getattr(settings, "mcp_server_url", None):
        return "http_gateway"
    if integration_id == "chrome" and bool(getattr(settings, "sdk_agent_enable_chrome_mcp", False)):
        return "stdio_bridge"
    if integration_id == "dbx" and bool(getattr(settings, "sdk_agent_enable_dbx_mcp", False)):
        return "stdio_bridge"
    if integration_id == "qq_mail" and bool(getattr(settings, "sdk_agent_enable_qq_mail_mcp", False)):
        return "stdio_bridge"
    if _load_registry_servers(settings):
        return "stdio_bridge"
    return None


def _load_registry_servers(settings: Any) -> tuple[Any, ...]:
    path = Path(getattr(settings, "mcp_registry_path", Path(__file__).resolve().parents[4] / "config" / "mcp_servers.json"))
    environment = dict(os.environ)
    # Settings loads .env for the API, while child MCP processes inherit only
    # the actual process environment. Inject feature flags without exposing
    # any secret values in logs or configuration snapshots.
    environment["JOBPILOT_SDK_AGENT_ENABLE_CHROME_MCP"] = "true" if getattr(settings, "sdk_agent_enable_chrome_mcp", False) else "false"
    environment["JOBPILOT_SDK_AGENT_ENABLE_DBX_MCP"] = "true" if getattr(settings, "sdk_agent_enable_dbx_mcp", False) else "false"
    environment["JOBPILOT_SDK_AGENT_ENABLE_QQ_MAIL_MCP"] = "true" if getattr(settings, "sdk_agent_enable_qq_mail_mcp", False) else "false"
    environment["JOBPILOT_QQ_MAIL_PYTHON"] = str(getattr(settings, "qq_mail_python", "") or "")
    environment["JOBPILOT_QQ_MAIL_API_ROOT"] = str(Path(__file__).resolve().parents[2])
    environment["JOBPILOT_QQ_MAIL_USERNAME"] = str(getattr(settings, "qq_mail_username", "") or "")
    auth_code = getattr(settings, "qq_mail_auth_code", None)
    environment["JOBPILOT_QQ_MAIL_AUTH_CODE"] = (
        auth_code.get_secret_value() if hasattr(auth_code, "get_secret_value") else str(auth_code or "")
    )
    try:
        return load_mcp_server_configs(path, environ=environment)
    except ValueError as exc:
        # Invalid declaration must make the integration unavailable instead of
        # silently constructing a partially trusted server definition.
        import logging

        logging.getLogger(__name__).error(
            "MCP registry configuration is invalid",
            extra={"error_type": type(exc).__name__},
        )
        return ()


__all__ = ["configured_mcp_tool_names", "configured_mcp_transport_label", "create_configured_mcp_client"]
