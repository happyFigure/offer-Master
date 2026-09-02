from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .models import run_summary


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self._root = root.expanduser().resolve()

    @property
    def root(self) -> Path:
        return self._root

    def save_run(self, record: Mapping[str, Any]) -> None:
        session_id = str(record.get("sessionId") or "").strip()
        run_id = str(record.get("runId") or "").strip()
        if not run_id:
            return
        session_key = self._safe_key(session_id or "anonymous")
        run_key = self._safe_key(run_id)
        run_path = self._run_path(session_key, run_key)
        payload = dict(record)
        payload["storage"] = {
            "sessionKey": session_key,
            "runKey": run_key,
            "runPath": str(run_path),
        }
        self._write_json(run_path, payload)
        self._write_json(self._run_index_path(run_key), {"sessionKey": session_key, "runKey": run_key})
        self._update_session_index(session_key, payload)
        for artifact in payload.get("artifacts") or []:
            if not isinstance(artifact, Mapping):
                continue
            artifact_id = str(artifact.get("artifactId") or "").strip()
            if not artifact_id:
                continue
            self._write_json(
                self._artifact_path(artifact_id),
                {
                    "sessionKey": session_key,
                    "runKey": run_key,
                    "artifact": dict(artifact),
                },
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run_key = self._safe_key(run_id)
        index = self._read_json(self._run_index_path(run_key))
        if not isinstance(index, Mapping):
            return None
        session_key = str(index.get("sessionKey") or "").strip()
        if not session_key:
            return None
        payload = self._read_json(self._run_path(session_key, run_key))
        return dict(payload) if isinstance(payload, Mapping) else None

    def get_session(self, session_id: str) -> dict[str, Any]:
        session_key = self._safe_key(session_id or "anonymous")
        payload = self._read_json(self._session_path(session_key))
        if isinstance(payload, Mapping):
            return dict(payload)
        return {"sessionId": session_id, "sessionKey": session_key, "runs": []}

    def list_session_artifacts(self, session_id: str, *, limit: int = 50) -> dict[str, Any]:
        session = self.get_session(session_id)
        runs = list(session.get("runs") or [])
        runs = runs[-max(1, int(limit)) :]
        artifacts: list[dict[str, Any]] = []
        for run in runs:
            if not isinstance(run, Mapping):
                continue
            run_record = self.get_run(str(run.get("runId") or ""))
            if not isinstance(run_record, Mapping):
                continue
            for artifact in run_record.get("artifacts") or []:
                if isinstance(artifact, Mapping):
                    artifacts.append(dict(artifact))
        return {**session, "artifacts": artifacts}

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        payload = self._read_json(self._artifact_path(artifact_id))
        if not isinstance(payload, Mapping):
            return None
        artifact = payload.get("artifact")
        return dict(artifact) if isinstance(artifact, Mapping) else None

    def resolve_artifact_file(self, artifact_id: str) -> Path | None:
        artifact = self.get_artifact(artifact_id)
        if not artifact:
            return None
        raw_path = str(artifact.get("path") or "").strip()
        raw_root = str(artifact.get("root") or "").strip()
        if not raw_path or not raw_root:
            return None
        path = Path(raw_path).expanduser().resolve()
        root = Path(raw_root).expanduser().resolve()
        if not _is_relative_to(path, root):
            return None
        if not path.exists() or not path.is_file():
            return None
        return path

    def _update_session_index(self, session_key: str, record: Mapping[str, Any]) -> None:
        path = self._session_path(session_key)
        session = self._read_json(path)
        if not isinstance(session, Mapping):
            session = {
                "schemaVersion": "agent.artifacts.session/v1",
                "sessionId": str(record.get("sessionId") or ""),
                "sessionKey": session_key,
                "runs": [],
            }
        runs = [item for item in session.get("runs") or [] if isinstance(item, Mapping)]
        summary = run_summary(record)
        runs = [item for item in runs if str(item.get("runId") or "") != summary["runId"]]
        runs.append(summary)
        session = {**dict(session), "runs": runs[-200:]}
        self._write_json(path, session)

    def _run_path(self, session_key: str, run_key: str) -> Path:
        return self._root / "runs" / session_key / f"{run_key}.json"

    def _session_path(self, session_key: str) -> Path:
        return self._root / "sessions" / f"{session_key}.json"

    def _run_index_path(self, run_key: str) -> Path:
        return self._root / "run-index" / f"{run_key}.json"

    def _artifact_path(self, artifact_id: str) -> Path:
        return self._root / "artifact-index" / f"{self._safe_key(artifact_id)}.json"

    @staticmethod
    def _safe_key(value: str) -> str:
        import hashlib

        digest = hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:16]
        text = "".join(ch for ch in str(value or "").strip() if ch.isalnum() or ch in {"-", "_"})
        if text:
            return f"{text[:72]}-{digest}"
        return f"sha256-{digest}"

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            Path(tmp_name).replace(path)
        finally:
            tmp_path = Path(tmp_name)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
