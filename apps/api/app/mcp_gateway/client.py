from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass(frozen=True)
class MCPToolCallResult:
    tool_name: str
    ok: bool
    result: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MCPGatewayClientProtocol(Protocol):
    def call_tool(self, *, tool_name: str, arguments: dict[str, Any]) -> MCPToolCallResult:
        raise NotImplementedError


class HttpMCPGatewayClient:
    def __init__(self, *, server_url: str, timeout_seconds: float = 30.0) -> None:
        self._server_url = server_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def call_tool(self, *, tool_name: str, arguments: dict[str, Any]) -> MCPToolCallResult:
        response = httpx.post(
            f"{self._server_url}/tools/{tool_name}/call",
            json={"arguments": arguments},
            timeout=self._timeout_seconds,
        )
        status_code = response.status_code
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return MCPToolCallResult(
                tool_name=tool_name,
                ok=True,
                result=payload,
                metadata={"status_code": status_code},
            )
        return MCPToolCallResult(
            tool_name=tool_name,
            ok=bool(payload.get("ok", True)),
            result=payload.get("result", payload),
            error=payload.get("error"),
            metadata={"status_code": status_code, **dict(payload.get("metadata") or {})},
        )
