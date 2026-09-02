from __future__ import annotations

import argparse
import os
import shutil

from cli_utils import resolve_path


def _clear_readonly_and_retry(func, path, exc_info):
    try:
        os.chmod(path, 0o666)
        func(path)
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete a file or directory.")
    parser.add_argument("path", nargs="?", help="Target path.")
    parser.add_argument("--path", dest="path_flag", help="Target path.")
    parser.add_argument("--recursive", action="store_true", help="Allow recursive deletion for directories.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow deletion without extra prompts.",
    )
    parser.add_argument(
        "--force-file",
        action="store_true",
        help="Allow file deletion when the target is a file.",
    )
    args, unknown = parser.parse_known_args()

    path = resolve_path(
        args.path or args.path_flag,
        ["--path", "-p", "--file-path", "--filepath", "--target"],
        unknown,
    )
    if path is None:
        print("ERROR: missing path argument")
        return
    if not path.exists():
        print(f"ERROR: path not found: {path}")
        return

    if path.is_dir():
        if not args.recursive:
            print("ERROR: directory deletion requires --recursive")
            return
        if not args.force:
            print("ERROR: directory deletion requires --force")
            return
        shutil.rmtree(path, onerror=_clear_readonly_and_retry)
        print(f"DELETED DIR: {path}")
        return

    if not (args.force or args.force_file):
        print("ERROR: file deletion requires --force or --force-file")
        return
    try:
        path.unlink()
    except PermissionError:
        os.chmod(path, 0o666)
        path.unlink()
    print(f"DELETED FILE: {path}")


if __name__ == "__main__":
    main()
