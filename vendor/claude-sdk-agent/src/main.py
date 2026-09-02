from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

try:
    import pwd
except ImportError:
    pwd = None

ROOT = Path(__file__).resolve().parents[1]

if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))
    from src.api.server import create_app
    from src.logging_setup import configure_logging
    from src.startup_permissions import repair_tree_permissions
else:
    from .api.server import create_app
    from .logging_setup import configure_logging
    from .startup_permissions import repair_tree_permissions


def _create_app():
    configure_logging(ROOT)
    return create_app(ROOT)


def _resolve_sudo_target():
    if pwd is None:
        return None
    sudo_uid = os.getenv("SUDO_UID")
    sudo_user = os.getenv("SUDO_USER")
    if sudo_uid:
        try:
            return pwd.getpwuid(int(sudo_uid))
        except Exception:
            pass
    if sudo_user:
        try:
            return pwd.getpwnam(sudo_user)
        except Exception:
            pass
    return None


def _repair_repo_permissions_for_sudo_user() -> None:
    if os.name == "nt" or pwd is None or not hasattr(os, "geteuid"):
        return
    if os.geteuid() != 0:
        return
    target = _resolve_sudo_target()
    if target is None:
        return
    repo_root = ROOT.parents[1]
    changed, skipped = repair_tree_permissions(repo_root, uid=target.pw_uid, gid=target.pw_gid)
    print(
        f"[claude-sdk-agent] repaired repo permissions root={repo_root} user={target.pw_name} changed={changed} skipped={skipped}",
        file=sys.stderr,
    )


def _drop_to_sudo_user() -> None:
    if os.name == "nt" or pwd is None or not hasattr(os, "geteuid"):
        return
    if os.geteuid() != 0:
        return
    if os.getenv("CLAUDE_SDK_AGENT_KEEP_ROOT", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    target = _resolve_sudo_target()
    if target is None:
        return
    try:
        os.initgroups(target.pw_name, target.pw_gid)
        os.setgid(target.pw_gid)
        os.setuid(target.pw_uid)
    except PermissionError:
        return
    os.environ["HOME"] = target.pw_dir
    os.environ["USER"] = target.pw_name
    os.environ["LOGNAME"] = target.pw_name
    os.environ.setdefault("SHELL", target.pw_shell or "/bin/bash")


app = None if __name__ == "__main__" else _create_app()


def main() -> int:
    parser = argparse.ArgumentParser(description="claude-sdk-agent service")
    parser.add_argument("mode", choices=["serve"], nargs="?", default="serve", help="run mode")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18008)
    args = parser.parse_args()

    _repair_repo_permissions_for_sudo_user()
    _drop_to_sudo_user()
    os.environ["CLAUDE_SDK_AGENT_HOST"] = str(args.host)
    os.environ["CLAUDE_SDK_AGENT_PORT"] = str(args.port)
    uvicorn.run(_create_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
