from __future__ import annotations

import argparse
from pathlib import Path

from cli_utils import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a directory.")
    parser.add_argument("path", nargs="?", help="Directory path.")
    parser.add_argument("--path", dest="path_flag", help="Directory path.")
    parser.add_argument("--exist-ok", action="store_true", help="Do not error if the directory exists.")
    args, unknown = parser.parse_known_args()

    path = resolve_path(
        args.path or args.path_flag,
        ["--path", "-p", "--dir", "--directory", "--target"],
        unknown,
    )
    if path is None:
        path = Path.cwd()
        print(f"INFO: missing path argument, defaulting to {path}")
        if path.exists():
            print(f"EXISTS: {path}")
            return
    if path.exists() and not args.exist_ok:
        print(f"ERROR: path already exists: {path}")
        return

    path.mkdir(parents=True, exist_ok=args.exist_ok)
    print(f"CREATED: {path}")


if __name__ == "__main__":
    main()
