from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from .config import SkillSettings

logger = logging.getLogger(__name__)


def sync_skill_mount(settings: SkillSettings) -> tuple[Path, list[str]]:
    mount_root = settings.mount_dir
    skills_root = mount_root / ".claude" / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)

    kept_names: set[str] = set()
    linked_names: list[str] = []

    for source_dir in settings.source_dirs:
        if not source_dir.exists() or not source_dir.is_dir():
            logger.warning("[skills] source directory missing: %s", source_dir)
            continue
        for entry in sorted(source_dir.iterdir(), key=lambda p: p.name):
            if not entry.is_dir():
                continue
            if not (entry / "SKILL.md").exists():
                continue
            if entry.name in kept_names:
                logger.info("[skills] skipped duplicate skill name=%s source=%s", entry.name, entry)
                continue
            target = skills_root / entry.name
            _replace_with_symlink(target, entry)
            kept_names.add(entry.name)
            linked_names.append(entry.name)

    for existing in skills_root.iterdir():
        if existing.name in kept_names:
            continue
        if existing.is_symlink() or existing.is_dir() or existing.exists():
            if existing.is_dir() and not existing.is_symlink():
                for child in existing.iterdir():
                    if child.is_dir():
                        for nested in child.rglob("*"):
                            pass
                _remove_path(existing)
            else:
                existing.unlink(missing_ok=True)

    logger.info("[skills] mounted count=%s root=%s names=%s", len(linked_names), skills_root, linked_names)
    return mount_root, linked_names


def _replace_with_symlink(target: Path, source: Path) -> None:
    if target.is_symlink():
        if Path(os.readlink(target)).resolve() == source.resolve():
            return
        target.unlink()
    elif target.exists():
        _remove_path(target)
    try:
        target.symlink_to(source, target_is_directory=True)
    except OSError as exc:
        logger.warning("[skills] symlink failed, copying skill source=%s target=%s error=%s", source, target, exc)
        shutil.copytree(source, target)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    if path.is_dir():
        for child in path.iterdir():
            _remove_path(child)
        path.rmdir()
