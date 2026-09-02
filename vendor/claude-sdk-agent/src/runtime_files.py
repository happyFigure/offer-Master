from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path

from .config import AppSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AuditLink:
    name: str
    path: str
    target: str
    status: str


def ensure_runtime_files(settings: AppSettings) -> None:
    allow_users_path = settings.auth.allow_users_path
    allow_users_path.parent.mkdir(parents=True, exist_ok=True)
    if not allow_users_path.exists():
        allow_users_path.write_text(
            json.dumps({"allow_users": []}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def ensure_audit_links(settings: AppSettings) -> list[AuditLink]:
    audit_dir = _audit_dir(settings)
    audit_dir.mkdir(parents=True, exist_ok=True)
    links: list[AuditLink] = []
    config_dir = settings.claude.config_dir
    links.append(_ensure_audit_link(audit_dir, "claude-config", config_dir))
    for name, target in _candidate_claude_audit_paths(config_dir).items():
        links.append(_ensure_audit_link(audit_dir, name, target))
    _write_audit_index(audit_dir, links)
    return links


def _audit_dir(settings: AppSettings) -> Path:
    return settings.root / "data" / "audit"


def _candidate_claude_audit_paths(config_dir: Path) -> dict[str, Path]:
    return {
        "claude-sessions": config_dir / "sessions",
        "claude-logs": config_dir / "logs",
        "claude-projects": config_dir / "projects",
        "claude-telemetry": config_dir / "telemetry",
        "claude-shell-snapshots": config_dir / "shell-snapshots",
    }


def _ensure_audit_link(audit_dir: Path, name: str, target: Path) -> AuditLink:
    link_path = audit_dir / name
    resolved_target = target.resolve()
    if not resolved_target.exists():
        _remove_stale_link(link_path)
        return AuditLink(name=name, path=str(link_path), target=str(resolved_target), status="missing")
    try:
        if link_path.is_symlink():
            current = Path(os.readlink(link_path))
            if not current.is_absolute():
                current = (link_path.parent / current).resolve()
            else:
                current = current.resolve()
            if current == resolved_target:
                return AuditLink(name=name, path=str(link_path), target=str(resolved_target), status="linked")
            link_path.unlink()
        elif link_path.exists():
            return AuditLink(name=name, path=str(link_path), target=str(resolved_target), status="blocked")
        link_path.symlink_to(resolved_target, target_is_directory=resolved_target.is_dir())
        return AuditLink(name=name, path=str(link_path), target=str(resolved_target), status="linked")
    except OSError as exc:
        logger.warning("[audit-links] failed name=%s path=%s target=%s err=%s", name, link_path, resolved_target, exc)
        return AuditLink(name=name, path=str(link_path), target=str(resolved_target), status="failed")


def _remove_stale_link(path: Path) -> None:
    try:
        if path.is_symlink() and not path.exists():
            path.unlink()
    except OSError as exc:
        logger.warning("[audit-links] failed to remove stale link path=%s err=%s", path, exc)


def _write_audit_index(audit_dir: Path, links: list[AuditLink]) -> None:
    payload = {
        "links": [
            {
                "name": item.name,
                "path": item.path,
                "target": item.target,
                "status": item.status,
            }
            for item in links
        ]
    }
    (audit_dir / "audit-links.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
