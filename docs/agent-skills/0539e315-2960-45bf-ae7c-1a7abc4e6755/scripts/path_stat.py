from __future__ import annotations

import argparse
from pathlib import Path

from cli_utils import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Show path stats.")
    parser.add_argument("path", nargs="?", help="Target path.")
    parser.add_argument("--path", dest="path_flag", help="Target path.")
    args, unknown = parser.parse_known_args()

    path = resolve_path(
        args.path or args.path_flag,
        ["--path", "-p", "--file-path", "--filepath", "--target"],
        unknown,
    )
    if path is None:
        path = Path.cwd()
        print(f"INFO: missing path argument, defaulting to {path}")
    if not path.exists():
        print(f"ERROR: path not found: {path}")
        return

    stat = path.stat()
    kind = "dir" if path.is_dir() else "file"
    print(f"PATH: {path}")
    print(f"TYPE: {kind}")
    print(f"SIZE: {stat.st_size}")
    print(f"MTIME: {stat.st_mtime}")


if __name__ == "__main__":
    main()
