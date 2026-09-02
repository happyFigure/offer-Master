from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from .models import SessionMapping


class SessionMappingStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    async def get(self, frontend_session_id: str) -> SessionMapping | None:
        async with self._lock:
            data = self._read_all()
        item = data.get(frontend_session_id)
        if not isinstance(item, dict):
            return None
        claude_session_id = str(item.get("claude_session_id") or "").strip()
        if not claude_session_id:
            return None
        raw_workspace_add_dirs = item.get("workspace_add_dirs") or item.get("workspaceAddDirs") or []
        workspace_add_dirs = raw_workspace_add_dirs if isinstance(raw_workspace_add_dirs, list) else []
        return SessionMapping(
            frontend_session_id=frontend_session_id,
            claude_session_id=claude_session_id,
            updated_at=float(item.get("updated_at") or 0.0),
            model=str(item.get("model") or "").strip(),
            workspace_cwd=str(item.get("workspace_cwd") or item.get("workspaceCwd") or "").strip(),
            workspace_add_dirs=[
                str(path).strip()
                for path in workspace_add_dirs
                if str(path or "").strip()
            ],
            workspace_source=str(item.get("workspace_source") or item.get("workspaceSource") or "").strip(),
            workspace_configured=bool(
                item.get("workspace_configured")
                if item.get("workspace_configured") is not None
                else item.get("workspaceConfigured", False)
            ),
        )

    async def put(
        self,
        frontend_session_id: str,
        claude_session_id: str,
        *,
        model: str = "",
        workspace_cwd: str = "",
        workspace_add_dirs: list[str] | None = None,
        workspace_source: str = "",
        workspace_configured: bool = False,
    ) -> SessionMapping:
        mapping = SessionMapping(
            frontend_session_id=frontend_session_id,
            claude_session_id=claude_session_id,
            updated_at=time.time(),
            model=model,
            workspace_cwd=str(workspace_cwd or "").strip(),
            workspace_add_dirs=list(
                dict.fromkeys(
                    str(path).strip()
                    for path in (workspace_add_dirs or [])
                    if str(path or "").strip()
                )
            ),
            workspace_source=str(workspace_source or "").strip(),
            workspace_configured=bool(workspace_configured),
        )
        async with self._lock:
            data = self._read_all()
            data[frontend_session_id] = mapping.to_dict()
            self._write_all(data)
        return mapping

    def _read_all(self) -> dict[str, object]:
        if not self._path.exists():
            return {}
        text = self._path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}

    def _write_all(self, data: dict[str, object]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self._path)
