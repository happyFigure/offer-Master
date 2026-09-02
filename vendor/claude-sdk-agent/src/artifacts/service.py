from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import ARTIFACT_SCHEMA_VERSION, ArtifactRoot
from .store import ArtifactStore


class ClaudeArtifactService:
    def __init__(self, store: ArtifactStore, *, runtime_root: Path) -> None:
        self._store = store
        self._runtime_root = runtime_root.expanduser().resolve()

    @property
    def store(self) -> ArtifactStore:
        return self._store

    async def save_run_from_paths(
        self,
        *,
        session_id: str,
        run_id: str,
        workspace_cwd: Path,
        workspace_add_dirs: Sequence[Path],
        affected_files: Sequence[str],
        started_at: float,
        status: str,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._save_run_from_paths_sync,
            session_id=session_id,
            run_id=run_id,
            workspace_cwd=workspace_cwd,
            workspace_add_dirs=tuple(workspace_add_dirs),
            affected_files=tuple(affected_files),
            started_at=started_at,
            status=status,
        )

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._store.get_run, run_id)

    async def list_session_artifacts(self, session_id: str, *, limit: int = 50) -> dict[str, Any]:
        return await asyncio.to_thread(self._store.list_session_artifacts, session_id, limit=limit)

    async def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._store.get_artifact, artifact_id)

    async def resolve_artifact_file(self, artifact_id: str) -> Path | None:
        return await asyncio.to_thread(self._store.resolve_artifact_file, artifact_id)

    def _save_run_from_paths_sync(
        self,
        *,
        session_id: str,
        run_id: str,
        workspace_cwd: Path,
        workspace_add_dirs: Sequence[Path],
        affected_files: Sequence[str],
        started_at: float,
        status: str,
    ) -> dict[str, Any]:
        completed_at = time.time()
        roots = self._build_roots(workspace_cwd, workspace_add_dirs)
        artifacts, errors = self._build_artifacts(
            session_id=session_id,
            run_id=run_id,
            roots=roots,
            paths=affected_files,
        )
        summary = {
            "created": sum(1 for item in artifacts if item["changeType"] == "created"),
            "modified": sum(1 for item in artifacts if item["changeType"] == "modified"),
            "deleted": sum(1 for item in artifacts if item["changeType"] == "deleted"),
            "artifactCount": len(artifacts),
            "truncated": False,
        }
        workspace_root = roots[0].path.as_posix() if roots else ""
        add_dirs = [root.path.as_posix() for root in roots[1:]]
        record = {
            "schemaVersion": ARTIFACT_SCHEMA_VERSION,
            "sessionId": str(session_id or ""),
            "runId": str(run_id or ""),
            "runtime": "claude-sdk-agent",
            "createdAt": float(started_at or completed_at),
            "completedAt": completed_at,
            "status": str(status or "completed"),
            "workspace": {"cwd": workspace_root, "addDirs": add_dirs},
            "runtimeRoot": self._runtime_root.as_posix(),
            "summary": summary,
            "artifacts": artifacts,
            "errors": errors[:50],
        }
        self._store.save_run(record)
        return record

    def _build_roots(self, workspace_cwd: Path, workspace_add_dirs: Sequence[Path]) -> list[ArtifactRoot]:
        roots: list[ArtifactRoot] = []
        seen: set[str] = set()
        for role, path in [("workspace", workspace_cwd), *(("add_dir", item) for item in workspace_add_dirs)]:
            try:
                resolved = Path(path).expanduser().resolve()
            except OSError:
                continue
            key = resolved.as_posix()
            if key in seen or not resolved.exists() or not resolved.is_dir():
                continue
            seen.add(key)
            roots.append(ArtifactRoot(resolved, role))
        return roots

    def _build_artifacts(
        self,
        *,
        session_id: str,
        run_id: str,
        roots: Sequence[ArtifactRoot],
        paths: Sequence[str],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        artifacts: list[dict[str, Any]] = []
        errors: list[str] = []
        seen_paths: set[str] = set()
        cwd = roots[0].path if roots else self._runtime_root
        for raw_path in paths:
            path = _resolve_affected_path(raw_path, cwd)
            if path is None:
                errors.append(f"invalid_path:{raw_path}")
                continue
            key = path.as_posix()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            root = _matching_root(path, roots)
            if root is None:
                errors.append(f"outside_workspace:{key}")
                continue
            if path.exists() and not path.is_file():
                errors.append(f"not_a_file:{key}")
                continue
            artifacts.append(
                self._artifact_item(
                    session_id=session_id,
                    run_id=run_id,
                    root=root,
                    path=path,
                )
            )
        return (
            sorted(artifacts, key=lambda item: (str(item.get("path") or ""), str(item.get("changeType") or ""))),
            errors,
        )

    def _artifact_item(
        self,
        *,
        session_id: str,
        run_id: str,
        root: ArtifactRoot,
        path: Path,
    ) -> dict[str, Any]:
        exists = path.exists() and path.is_file()
        stat = path.stat() if exists else None
        change_type = "modified" if exists else "deleted"
        try:
            relative_path = path.relative_to(root.path).as_posix()
        except ValueError:
            relative_path = path.name
        artifact_id = self._artifact_id(
            session_id=session_id,
            run_id=run_id,
            path=path,
            change_type=change_type,
        )
        return {
            "artifactId": artifact_id,
            "sessionId": str(session_id or ""),
            "runId": str(run_id or ""),
            "path": path.as_posix(),
            "relativePath": relative_path,
            "root": root.path.as_posix(),
            "rootRole": root.role,
            "kind": "file",
            "changeType": change_type,
            "source": "sdk_affected_files",
            "confidence": "high" if exists else "medium",
            "size": int(stat.st_size) if stat else 0,
            "mtimeNs": int(stat.st_mtime_ns) if stat else 0,
            "mode": int(stat.st_mode) if stat else 0,
            "mimeType": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            "downloadable": exists,
            "openable": exists,
            "availableActions": ["open", "download"] if exists else [],
        }

    @staticmethod
    def _artifact_id(
        *,
        session_id: str,
        run_id: str,
        path: Path,
        change_type: str,
    ) -> str:
        digest = hashlib.sha256(
            f"{session_id}\0{run_id}\0{change_type}\0{path.as_posix()}".encode("utf-8")
        ).hexdigest()[:24]
        return f"art_{digest}"


def _resolve_affected_path(raw_path: str, cwd: Path) -> Path | None:
    text = str(raw_path or "").replace("\x00", "").strip()
    if not text:
        return None
    try:
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = cwd / path
        return path.resolve()
    except OSError:
        return None


def _matching_root(path: Path, roots: Sequence[ArtifactRoot]) -> ArtifactRoot | None:
    matches = [root for root in roots if _is_relative_to(path, root.path)]
    if not matches:
        return None
    return max(matches, key=lambda root: len(root.path.parts))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
