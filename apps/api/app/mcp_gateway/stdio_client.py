from __future__ import annotations

import atexit
from concurrent.futures import Future
from dataclasses import dataclass, field
import logging
import queue
import threading
from typing import Any, Protocol

from app.mcp_gateway.client import MCPToolCallResult


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPStdioServerSpec:
    tool_prefix: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        prefix = str(self.tool_prefix or "").strip().strip(".").lower()
        if not prefix:
            raise ValueError("MCP stdio server tool_prefix is required")
        command = str(self.command or "").strip()
        if not command:
            raise ValueError(f"MCP stdio command is required: {prefix}")
        object.__setattr__(self, "tool_prefix", prefix)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "args", tuple(str(arg) for arg in self.args))
        object.__setattr__(self, "env", {str(key): str(value) for key, value in dict(self.env or {}).items()})
        object.__setattr__(self, "timeout_seconds", max(1.0, float(self.timeout_seconds or 30.0)))


class StdioMCPTransport(Protocol):
    async def list_tools(self, server: MCPStdioServerSpec) -> list[dict[str, Any]]:
        ...

    async def call_tool(self, server: MCPStdioServerSpec, *, tool_name: str, arguments: dict[str, Any]) -> MCPToolCallResult:
        ...


class _PersistentServerSession:
    """Own one long-lived MCP stdio process and serialize its requests.

    MCP servers such as Chrome DevTools keep useful state in the server
    process. Starting a fresh subprocess for every tool call would discard
    the selected page and browser state, so the runtime keeps one session per
    declared server spec and reuses it across discovery and calls.
    """

    def __init__(self, server: MCPStdioServerSpec) -> None:
        self.server = server
        self._requests: queue.Queue[tuple[str, dict[str, Any], Future[Any]] | None] = queue.Queue()
        self._ready = threading.Event()
        self._stop_requested = threading.Event()
        self._start_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"offermaster-mcp-{server.tool_prefix}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Started persistent MCP stdio session", extra={"mcp_prefix": server.tool_prefix})

    def submit(self, operation: str, **payload: Any) -> Any:
        if not self._ready.wait(timeout=self.server.timeout_seconds):
            raise TimeoutError(f"MCP server did not initialize: {self.server.tool_prefix}")
        if self._start_error is not None:
            raise self._start_error
        if self._stop_requested.is_set():
            raise RuntimeError(f"MCP server session is closed: {self.server.tool_prefix}")
        future: Future[Any] = Future()
        self._requests.put((operation, payload, future))
        return future.result(timeout=self.server.timeout_seconds)

    def close(self) -> None:
        if self._stop_requested.is_set():
            return
        self._stop_requested.set()
        self._requests.put(None)
        self._thread.join(timeout=min(max(self.server.timeout_seconds, 1.0), 5.0))
        logger.info("Closed persistent MCP stdio session", extra={"mcp_prefix": self.server.tool_prefix})

    def _run(self) -> None:
        try:
            import anyio

            anyio.run(self._run_async)
        except BaseException as exc:  # pragma: no cover - process failures vary by MCP server.
            self._start_error = exc
            self._ready.set()
            self._fail_pending(exc)
            logger.warning(
                "Persistent MCP stdio session stopped",
                extra={"mcp_prefix": self.server.tool_prefix, "error_type": type(exc).__name__},
            )

    async def _run_async(self) -> None:
        import anyio

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.server.command,
            args=list(self.server.args),
            env=dict(self.server.env) or None,
            cwd=self.server.cwd,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=self.server.timeout_seconds) as session:
                await session.initialize()
                self._ready.set()
                while not self._stop_requested.is_set():
                    request = await anyio.to_thread.run_sync(self._requests.get)
                    if request is None:
                        break
                    operation, payload, future = request
                    try:
                        if operation == "list_tools":
                            result = await session.list_tools()
                        else:
                            result = await session.call_tool(
                                str(payload["tool_name"]),
                                dict(payload.get("arguments") or {}),
                            )
                        future.set_result(result)
                    except BaseException as exc:  # Keep the worker alive for the next request.
                        future.set_exception(exc)

    def _fail_pending(self, error: BaseException) -> None:
        while True:
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                return
            if request is not None:
                request[2].set_exception(error)


class PersistentStdioMCPTransport:
    """Reuse one MCP process per server spec across sync runtime calls."""

    def __init__(self) -> None:
        self._sessions: dict[tuple[Any, ...], _PersistentServerSession] = {}
        self._lock = threading.Lock()

    async def list_tools(self, server: MCPStdioServerSpec) -> list[dict[str, Any]]:
        session = self._get_session(server)
        result = await _run_in_worker_thread(lambda: session.submit("list_tools"))
        payload = _model_dump(result)
        return [item for item in (_model_dump(tool) for tool in payload.get("tools") or []) if item.get("name")]

    async def call_tool(self, server: MCPStdioServerSpec, *, tool_name: str, arguments: dict[str, Any]) -> MCPToolCallResult:
        session = self._get_session(server)
        try:
            result = await _run_in_worker_thread(
                lambda: session.submit("call_tool", tool_name=tool_name, arguments=dict(arguments or {}))
            )
        except Exception as exc:  # pragma: no cover - exact process failures vary by MCP server.
            logger.warning(
                "Persistent MCP stdio tool call failed",
                extra={"mcp_prefix": server.tool_prefix, "mcp_tool_name": tool_name, "error_type": type(exc).__name__},
            )
            return MCPToolCallResult(
                tool_name=tool_name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                result={"message": "MCP stdio server call failed."},
            )
        return _mcp_call_result(tool_name=tool_name, result=result)

    def close(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()

    def _get_session(self, server: MCPStdioServerSpec) -> _PersistentServerSession:
        key = (
            server.tool_prefix,
            server.command,
            server.args,
            tuple(sorted(server.env.items())),
            server.cwd,
        )
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                session = _PersistentServerSession(server)
                self._sessions[key] = session
            else:
                logger.debug("Reusing persistent MCP stdio session", extra={"mcp_prefix": server.tool_prefix})
            return session


async def _run_in_worker_thread(factory: Any) -> Any:
    import anyio

    return await anyio.to_thread.run_sync(factory)


_DEFAULT_STDIO_TRANSPORT = PersistentStdioMCPTransport()
atexit.register(_DEFAULT_STDIO_TRANSPORT.close)


class RealStdioMCPTransport:
    async def list_tools(self, server: MCPStdioServerSpec) -> list[dict[str, Any]]:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            return []

        params = StdioServerParameters(
            command=server.command,
            args=list(server.args),
            env=dict(server.env) or None,
            cwd=server.cwd,
        )
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=server.timeout_seconds) as session:
                    await session.initialize()
                    result = await session.list_tools()
        except Exception as exc:  # pragma: no cover - exact server failures vary.
            logger.warning(
                "MCP stdio tool discovery failed",
                extra={"mcp_prefix": server.tool_prefix, "error_type": type(exc).__name__},
            )
            return []
        payload = _model_dump(result)
        raw_tools = payload.get("tools") or []
        return [item for item in (_model_dump(tool) for tool in raw_tools) if item.get("name")]

    async def call_tool(self, server: MCPStdioServerSpec, *, tool_name: str, arguments: dict[str, Any]) -> MCPToolCallResult:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:  # pragma: no cover - depends on optional runtime package.
            return MCPToolCallResult(
                tool_name=tool_name,
                ok=False,
                error="MCP_CLIENT_NOT_INSTALLED",
                result={"message": str(exc)},
            )

        params = StdioServerParameters(
            command=server.command,
            args=list(server.args),
            env=dict(server.env) or None,
            cwd=server.cwd,
        )
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=server.timeout_seconds) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
        except Exception as exc:  # pragma: no cover - exact MCP transport failures vary by server.
            logger.exception(
                "MCP stdio tool call failed",
                extra={"mcp_prefix": server.tool_prefix, "mcp_tool_name": tool_name},
            )
            return MCPToolCallResult(
                tool_name=tool_name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                result={"message": "MCP stdio server call failed."},
            )
        return _mcp_call_result(tool_name=tool_name, result=result)


class StdioMCPGatewayClient:
    """Small adapter from OfferMaster's sync tool handlers to stdio MCP servers.

    The runtime registry still exposes explicit allowlisted names such as
    ``chrome.list_pages``. This adapter only strips the first namespace segment
    and sends the remaining concrete tool name to the matching MCP server.
    """

    def __init__(self, *, servers: tuple[MCPStdioServerSpec, ...], transport: StdioMCPTransport | None = None) -> None:
        self._servers = {server.tool_prefix: server for server in servers}
        self._transport = transport or _DEFAULT_STDIO_TRANSPORT

    def list_tools(self, *, prefix: str | None = None) -> list[dict[str, Any]]:
        """Discover MCP schemas and namespace them for the OfferMaster registry."""

        servers = self._servers.values() if prefix is None else [self._servers.get(str(prefix).lower())]
        discovered: list[dict[str, Any]] = []
        for server in servers:
            if server is None:
                continue
            raw_tools = _run_async_blocking(lambda server=server: self._transport.list_tools(server))
            for tool in raw_tools or []:
                tool_name = str(tool.get("name") or "").strip()
                if not tool_name:
                    continue
                discovered.append(
                    {
                        "name": f"{server.tool_prefix}.{tool_name}",
                        "server_tool_name": tool_name,
                        "server_prefix": server.tool_prefix,
                        "description": str(tool.get("description") or "Call MCP tool: " + tool_name),
                        "inputSchema": dict(tool.get("inputSchema") or tool.get("input_schema") or {"type": "object"}),
                    }
                )
        return discovered

    def call_tool(self, *, tool_name: str, arguments: dict[str, Any]) -> MCPToolCallResult:
        resolved = self._resolve_tool(tool_name)
        if resolved is None:
            return MCPToolCallResult(
                tool_name=tool_name,
                ok=False,
                error="MCP_SERVER_NOT_CONFIGURED",
                result={"message": f"No stdio MCP server configured for tool: {tool_name}"},
            )
        server, server_tool_name = resolved
        logger.info(
            "Dispatching MCP tool through stdio bridge",
            extra={"mcp_prefix": server.tool_prefix, "mcp_tool_name": server_tool_name},
        )
        result = _run_async_blocking(lambda: self._transport.call_tool(server, tool_name=server_tool_name, arguments=dict(arguments or {})))
        return MCPToolCallResult(
            tool_name=tool_name,
            ok=result.ok,
            result=result.result,
            error=result.error,
            metadata={"mcp_prefix": server.tool_prefix, "mcp_tool_name": server_tool_name, **dict(result.metadata or {})},
        )

    def _resolve_tool(self, tool_name: str) -> tuple[MCPStdioServerSpec, str] | None:
        normalized = str(tool_name or "").strip()
        if "." not in normalized:
            return None
        prefix, server_tool_name = normalized.split(".", 1)
        server = self._servers.get(prefix.lower())
        if server is None or not server_tool_name:
            return None
        return server, server_tool_name


def _run_async_blocking(async_factory: Any) -> Any:
    """Run an async MCP call from sync code, even when the SDK tool loop is async."""

    box: dict[str, Any] = {}

    def runner() -> None:
        try:
            import anyio

            box["result"] = anyio.run(async_factory)
        except BaseException as exc:  # pragma: no cover - re-raised in caller thread.
            box["error"] = exc

    thread = threading.Thread(target=runner, name="offermaster-mcp-stdio", daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("result")


def _mcp_call_result(*, tool_name: str, result: Any) -> MCPToolCallResult:
    payload = _model_dump(result)
    is_error = bool(payload.get("is_error", False))
    return MCPToolCallResult(
        tool_name=tool_name,
        ok=not is_error,
        result=_result_payload(payload),
        error="MCP_TOOL_ERROR" if is_error else None,
        metadata={"result_type": payload.get("result_type")},
    )


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if isinstance(value, dict):
        return dict(value)
    return {"value": value}


def _result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    structured = payload.get("structured_content")
    if structured is not None:
        return {"structured_content": structured, "content": payload.get("content") or []}
    return {"content": payload.get("content") or [], "text": _text_from_content(payload.get("content"))}


def _text_from_content(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = str(item.get("text") or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


__all__ = [
    "MCPStdioServerSpec",
    "PersistentStdioMCPTransport",
    "RealStdioMCPTransport",
    "StdioMCPGatewayClient",
    "StdioMCPTransport",
]
