from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping


@dataclass(slots=True)
class SessionClientRecord:
    frontend_session_id: str
    claude_session_id: str
    model: str
    resumed: bool
    signature: str
    client: Any
    workspace_cwd: str = ""
    workspace_add_dirs: list[str] = field(default_factory=list)
    workspace_source: str = ""
    workspace_configured: bool = False
    workspace_runtime: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    server_info: Mapping[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "frontendSessionId": self.frontend_session_id,
            "claudeSessionId": self.claude_session_id,
            "model": self.model,
            "resumed": self.resumed,
            "signatureDigest": hashlib.sha256(self.signature.encode("utf-8")).hexdigest()[:24],
            "workspaceCwd": self.workspace_cwd,
            "workspaceAddDirs": list(self.workspace_add_dirs),
            "workspaceSource": self.workspace_source,
            "workspaceConfigured": self.workspace_configured,
            "workspaceRuntime": dict(self.workspace_runtime),
            "createdAt": self.created_at,
            "lastUsedAt": self.last_used_at,
            "connected": True,
        }


class ClaudeClientPool:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._items: dict[str, SessionClientRecord] = {}

    async def get(self, frontend_session_id: str) -> SessionClientRecord | None:
        async with self._lock:
            record = self._items.get(frontend_session_id)
            if record is None:
                return None
            record.last_used_at = time.time()
            return record

    async def get_or_create(
        self,
        frontend_session_id: str,
        *,
        claude_session_id: str,
        model: str,
        resumed: bool,
        signature: str,
        workspace_cwd: str = "",
        workspace_add_dirs: list[str] | None = None,
        workspace_source: str = "",
        workspace_configured: bool = False,
        workspace_runtime: Mapping[str, Any] | None = None,
        factory: Callable[[], Awaitable[tuple[Any, Mapping[str, Any] | None]]],
    ) -> SessionClientRecord:
        stale: SessionClientRecord | None = None
        async with self._lock:
            existing = self._items.get(frontend_session_id)
            if existing is not None and existing.signature == signature:
                existing.last_used_at = time.time()
                return existing
            stale = existing
            if stale is not None:
                self._items.pop(frontend_session_id, None)
        if stale is not None:
            async with stale.lock:
                disconnect = getattr(stale.client, "disconnect", None)
                if callable(disconnect):
                    await disconnect()
        client, server_info = await factory()
        record = SessionClientRecord(
            frontend_session_id=frontend_session_id,
            claude_session_id=claude_session_id,
            model=model,
            resumed=resumed,
            signature=signature,
            client=client,
            workspace_cwd=str(workspace_cwd or "").strip(),
            workspace_add_dirs=list(workspace_add_dirs or []),
            workspace_source=str(workspace_source or "").strip(),
            workspace_configured=bool(workspace_configured),
            workspace_runtime=dict(workspace_runtime or {}),
            server_info=server_info,
        )
        async with self._lock:
            existing = self._items.get(frontend_session_id)
            if existing is not None and existing.signature == signature:
                disconnect = getattr(client, "disconnect", None)
                if callable(disconnect):
                    await disconnect()
                existing.last_used_at = time.time()
                return existing
            self._items[frontend_session_id] = record
            return record

    async def disconnect(self, frontend_session_id: str) -> bool:
        async with self._lock:
            record = self._items.pop(frontend_session_id, None)
        if record is None:
            return False
        async with record.lock:
            disconnect = getattr(record.client, "disconnect", None)
            if callable(disconnect):
                await disconnect()
        return True

    async def remove(self, frontend_session_id: str) -> SessionClientRecord | None:
        async with self._lock:
            return self._items.pop(frontend_session_id, None)

    async def disconnect_all(self) -> None:
        async with self._lock:
            items = list(self._items.values())
            self._items.clear()
        for record in items:
            async with record.lock:
                disconnect = getattr(record.client, "disconnect", None)
                if callable(disconnect):
                    await disconnect()

    async def runtime_snapshot(self, *, include_sessions: bool = False) -> Mapping[str, Any]:
        async with self._lock:
            records = list(self._items.values())
        payload: dict[str, Any] = {
            "connectedSessions": len(records),
        }
        if include_sessions:
            payload["sessions"] = [record.snapshot() for record in records]
        return payload
