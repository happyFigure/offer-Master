from __future__ import annotations

import os
from pathlib import Path
import stat


def repair_tree_permissions(root: Path, *, uid: int, gid: int) -> tuple[int, int]:
    if os.name == "nt":
        return (0, 0)
    if os.getenv("CLAUDE_SDK_AGENT_SKIP_REPO_PERMISSION_REPAIR", "").strip().lower() in {"1", "true", "yes", "on"}:
        return (0, 0)
    target_root = Path(root).resolve()
    if not target_root.exists():
        return (0, 0)

    changed = 0
    skipped = 0
    for path in _iter_paths(target_root):
        try:
            if path.is_symlink():
                skipped += 1
                continue
            current = path.stat()
            desired_mode = _desired_mode(path, current.st_mode)
            _safe_chown(path, uid=uid, gid=gid)
            _safe_chmod(path, desired_mode)
            changed += 1
        except OSError:
            skipped += 1
    return (changed, skipped)


def _iter_paths(root: Path):
    yield root
    for dirpath, dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)
        for name in dirnames:
            yield dir_path / name
        for name in filenames:
            yield dir_path / name


def _desired_mode(path: Path, current_mode: int) -> int:
    current = stat.S_IMODE(current_mode)
    if path.is_dir():
        return current | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
    return current | stat.S_IRUSR | stat.S_IWUSR


def _safe_chown(path: Path, *, uid: int, gid: int) -> None:
    try:
        os.chown(path, uid, gid, follow_symlinks=False)
    except NotImplementedError:
        os.chown(path, uid, gid)


def _safe_chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode, follow_symlinks=False)
    except NotImplementedError:
        os.chmod(path, mode)
