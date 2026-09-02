from __future__ import annotations

import argparse
import shutil

from cli_utils import get_flag_value, get_nth_positional, resolve_path


def _resolve_effective_dst(src, dst):
    if dst.exists() and dst.is_dir():
        return dst / src.name
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Move a file or directory.")
    parser.add_argument("src", nargs="?", help="Source path.")
    parser.add_argument("dst", nargs="?", help="Destination path.")
    parser.add_argument("--src", dest="src_flag", help="Source path.")
    parser.add_argument("--dst", dest="dst_flag", help="Destination path.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwrite if destination exists.")
    args, unknown = parser.parse_known_args()

    src = resolve_path(args.src or args.src_flag, [], [])
    dst = resolve_path(args.dst or args.dst_flag, [], [])
    if src is None:
        src = resolve_path(get_flag_value(unknown, ["--src", "--source", "--from"]), [], [])
    if dst is None:
        dst = resolve_path(get_flag_value(unknown, ["--dst", "--dest", "--destination", "--to", "--target"]), [], [])
    if src is None:
        src = resolve_path(get_nth_positional(unknown, 0), [], [])
    if dst is None:
        dst = resolve_path(get_nth_positional(unknown, 1), [], [])
    if src is None or dst is None:
        print("ERROR: missing src/dst arguments")
        return
    if not src.exists():
        print(f"ERROR: source not found: {src}")
        return

    effective_dst = _resolve_effective_dst(src, dst)
    if effective_dst.exists() and not args.overwrite:
        print(f"ERROR: destination exists (use --overwrite to replace): {effective_dst}")
        return

    if effective_dst.exists() and args.overwrite:
        if effective_dst.is_dir():
            shutil.rmtree(effective_dst)
        else:
            effective_dst.unlink()

    effective_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(effective_dst))
    print(f"MOVED: {src} -> {effective_dst}")


if __name__ == "__main__":
    main()
