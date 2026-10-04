from __future__ import annotations

import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


SDK_AGENT_SANDBOX_MODE_TEMP_COPY = "temp_copy"


class SandboxPathError(ValueError):
    """Raised when a child-agent file path tries to escape its sandbox."""


@dataclass(frozen=True)
class SdkAgentSandboxWorkspace:
    run_id: str
    mode: str
    root: Path
    input_dir: Path
    work_dir: Path
    output_dir: Path
    logical_inputs: dict[str, str]

    @classmethod
    def from_metadata(cls, payload: Mapping[str, Any]) -> "SdkAgentSandboxWorkspace":
        root = Path(str(payload["root"])).resolve()
        return cls(
            run_id=str(payload.get("run_id") or root.name),
            mode=str(payload.get("mode") or SDK_AGENT_SANDBOX_MODE_TEMP_COPY),
            root=root,
            input_dir=Path(str(payload.get("input_dir") or root / "input")).resolve(),
            work_dir=Path(str(payload.get("work_dir") or root / "work")).resolve(),
            output_dir=Path(str(payload.get("output_dir") or root / "output")).resolve(),
            logical_inputs=dict(payload.get("logical_inputs") or {}),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "root": str(self.root),
            "input_dir": str(self.input_dir),
            "work_dir": str(self.work_dir),
            "output_dir": str(self.output_dir),
            "logical_inputs": dict(self.logical_inputs),
        }

    def to_public_metadata(self, *, artifacts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "input_paths": sorted(self.logical_inputs),
            "work_dir": "work",
            "output_dir": "output",
            "artifacts": list(artifacts or []),
        }


@dataclass(frozen=True)
class SdkAgentSandboxManager:
    base_dir: Path | str
    mode: str = SDK_AGENT_SANDBOX_MODE_TEMP_COPY

    def prepare_workspace(
        self,
        *,
        run_id: str | None = None,
        inputs: Mapping[str, str | Path] | None = None,
    ) -> SdkAgentSandboxWorkspace:
        if self.mode != SDK_AGENT_SANDBOX_MODE_TEMP_COPY:
            raise ValueError(f"Unsupported SDK agent sandbox mode: {self.mode}")

        workspace_run_id = str(run_id or f"run_{uuid.uuid4().hex}")
        base_dir = Path(self.base_dir).resolve()
        root = (base_dir / workspace_run_id).resolve()
        if not _is_relative_to(root, base_dir):
            raise SandboxPathError("Sandbox run id resolves outside the sandbox base directory.")
        input_dir = root / "input"
        work_dir = root / "work"
        output_dir = root / "output"
        for directory in (input_dir, work_dir, output_dir):
            directory.mkdir(parents=True, exist_ok=True)

        logical_inputs: dict[str, str] = {}
        for logical_path, source_path in (inputs or {}).items():
            normalized = _normalize_logical_path(logical_path)
            if not normalized.startswith("input/"):
                raise SandboxPathError("Sandbox input files must be mounted under input/.")
            destination = _resolve_inside_root(root, normalized)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_input(Path(source_path), destination)
            logical_inputs[normalized] = str(source_path)

        return SdkAgentSandboxWorkspace(
            run_id=workspace_run_id,
            mode=self.mode,
            root=root,
            input_dir=input_dir,
            work_dir=work_dir,
            output_dir=output_dir,
            logical_inputs=logical_inputs,
        )

    def resolve_path(self, workspace: SdkAgentSandboxWorkspace, logical_path: str, *, access: str = "read") -> Path:
        normalized = _normalize_logical_path(logical_path)
        _validate_access(normalized, access=access)
        # Resolve after joining so traversal like "../" cannot escape the sandbox root.
        return _resolve_inside_root(workspace.root, normalized)

    def collect_artifacts(self, workspace: SdkAgentSandboxWorkspace) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []
        if not workspace.output_dir.exists():
            return artifacts
        for path in sorted(item for item in workspace.output_dir.rglob("*") if item.is_file()):
            logical_path = _as_posix_relative(path, workspace.root)
            artifacts.append(
                {
                    "logical_path": logical_path,
                    "size_bytes": path.stat().st_size,
                    "kind": path.suffix.lstrip(".").lower() or "file",
                }
            )
        return artifacts

    def cleanup(self, workspace: SdkAgentSandboxWorkspace) -> None:
        shutil.rmtree(workspace.root, ignore_errors=True)


def _copy_input(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(str(source))
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return
    shutil.copy2(source, destination)


def _normalize_logical_path(logical_path: str | Path) -> str:
    raw = str(logical_path or "").strip()
    if not raw:
        raise SandboxPathError("Sandbox path cannot be empty.")
    if "\x00" in raw:
        raise SandboxPathError("Sandbox path cannot contain null bytes.")
    if _is_absolute_or_drive_path(raw):
        raise SandboxPathError("Sandbox path must be relative to the sandbox root.")
    normalized = raw.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise SandboxPathError("Sandbox path cannot contain empty, current, or parent directory segments.")
    return pure.as_posix()


def _is_absolute_or_drive_path(path: str) -> bool:
    windows_path = PureWindowsPath(path)
    return bool(windows_path.drive or windows_path.root or PurePosixPath(path).is_absolute())


def _validate_access(normalized_path: str, *, access: str) -> None:
    operation = str(access or "read").strip().lower()
    top_level = normalized_path.split("/", 1)[0]
    if top_level not in {"input", "work", "output"}:
        raise SandboxPathError("Sandbox paths must start with input/, work/, or output/.")
    if operation in {"write", "replace", "delete", "move", "mkdir"} and top_level == "input":
        raise SandboxPathError("Sandbox input files are read-only for child agents.")


def _resolve_inside_root(root: Path, normalized_path: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root_resolved / normalized_path).resolve(strict=False)
    if not _is_relative_to(candidate, root_resolved):
        raise SandboxPathError("Sandbox path resolves outside the sandbox root.")
    return candidate


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _as_posix_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


__all__ = [
    "SDK_AGENT_SANDBOX_MODE_TEMP_COPY",
    "SandboxPathError",
    "SdkAgentSandboxManager",
    "SdkAgentSandboxWorkspace",
]
