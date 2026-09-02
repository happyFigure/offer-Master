from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from .config import WorkflowSettings
logger = logging.getLogger(__name__)


def sync_workflow_mount(settings: WorkflowSettings) -> tuple[Path, list[str]]:
    target_root = settings.target_dir
    target_root.mkdir(parents=True, exist_ok=True)

    kept_names: set[str] = set()
    linked_names: list[str] = []

    for source_dir in settings.source_dirs:
        if not source_dir.exists() or not source_dir.is_dir():
            logger.info("[workflows] source directory missing: %s", source_dir)
            continue
        for entry in sorted(source_dir.iterdir(), key=lambda p: p.name):
            if not _is_workflow_entry(entry):
                continue
            if entry.name in kept_names:
                logger.info("[workflows] skipped duplicate workflow name=%s source=%s", entry.name, entry)
                continue
            target = target_root / entry.name
            if not _replace_with_symlink(target, entry):
                logger.warning("[workflows] skipped existing non-managed workflow name=%s target=%s", entry.name, target)
                continue
            kept_names.add(entry.name)
            linked_names.append(entry.name)

    for existing in target_root.iterdir():
        if existing.name in kept_names:
            continue
        if existing.is_symlink() and _points_inside_any(existing, settings.source_dirs):
            existing.unlink()

    logger.info("[workflows] mounted count=%s root=%s names=%s", len(linked_names), target_root, linked_names)
    return target_root, linked_names


def _is_workflow_entry(path: Path) -> bool:
    if path.name.startswith("."):
        return False
    return path.is_file() or path.is_dir()


def _replace_with_symlink(target: Path, source: Path) -> bool:
    if target.is_symlink():
        if Path(os.readlink(target)).resolve() == source.resolve():
            return True
        target.unlink()
    elif target.exists():
        return False
    try:
        target.symlink_to(source, target_is_directory=source.is_dir())
    except OSError as exc:
        logger.warning("[workflows] symlink failed, copying source=%s target=%s error=%s", source, target, exc)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    return True


def _points_inside_any(target: Path, source_dirs: list[Path]) -> bool:
    try:
        resolved = target.resolve(strict=False)
    except OSError:
        return False
    for source_dir in source_dirs:
        try:
            resolved.relative_to(source_dir.resolve(strict=False))
            return True
        except ValueError:
            continue
    return False
