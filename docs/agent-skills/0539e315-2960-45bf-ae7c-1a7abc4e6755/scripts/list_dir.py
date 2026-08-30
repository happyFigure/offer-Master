from __future__ import annotations

import argparse
from pathlib import Path

from cli_utils import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="List directory contents.")
    parser.add_argument("path", nargs="?", help="Absolute directory path.")
    parser.add_argument("--path", dest="path_flag", help="Absolute directory path.")
    args, unknown = parser.parse_known_args()

    path = resolve_path(
        args.path or args.path_flag,
        ["--path", "-p", "--file-path", "--filepath", "--dir", "--directory", "--target"],
        unknown,
    )
    if path is None:
        path = Path.cwd()
    if not path.exists():
        print(f"ERROR: path not found: {path}")
        return
    if not path.is_dir():
        print(f"ERROR: path is not a directory: {path}")
        return

    for item in sorted(path.iterdir()):
        kind = "dir" if item.is_dir() else "file"
        print(f"{kind}\t{item.name}")


if __name__ == "__main__":
    main()
