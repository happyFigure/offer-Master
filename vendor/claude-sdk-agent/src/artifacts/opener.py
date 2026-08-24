from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


class ArtifactOpenError(RuntimeError):
    pass


def open_file_with_default_app(path: Path) -> None:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(str(resolved))

    if sys.platform.startswith("win"):
        os.startfile(str(resolved))  # type: ignore[attr-defined]
        return

    if sys.platform == "darwin":
        _spawn(["open", str(resolved)])
        return

    opener = shutil.which("xdg-open")
    if opener:
        _spawn([opener, str(resolved)])
        return

    gio = shutil.which("gio")
    if gio:
        _spawn([gio, "open", str(resolved)])
        return

    raise ArtifactOpenError("No supported local file opener found")


def _spawn(command: list[str]) -> None:
    try:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        raise ArtifactOpenError(str(exc)) from exc
