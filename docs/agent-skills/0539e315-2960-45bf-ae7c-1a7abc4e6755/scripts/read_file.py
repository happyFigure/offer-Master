from __future__ import annotations

import argparse
from pathlib import Path

from cli_utils import resolve_path


def _decode_content(data: bytes, encoding: str) -> str:
    if not encoding or encoding.lower() in {"auto", "detect"}:
        for candidate in ("utf-8-sig", "utf-8", "gb18030", "utf-16", "latin-1"):
            try:
                return data.decode(candidate)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")
    try:
        return data.decode(encoding)
    except LookupError:
        return data.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        return data.decode(encoding, errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a local file safely.")
    parser.add_argument("path", nargs="?", help="Absolute file path.")
    parser.add_argument("--path", dest="path_flag", help="Absolute file path.")
    parser.add_argument("--offset", type=int, default=0, help="Start line offset (0-based).")
    parser.add_argument("--limit", type=int, default=0, help="Max number of lines to read (0 = all).")
    parser.add_argument("--encoding", default="auto", help="Text encoding.")
    args, unknown = parser.parse_known_args()

    path = resolve_path(
        args.path or args.path_flag,
        ["--path", "-p", "--file-path", "--filepath", "--target"],
        unknown,
    )
    if path is None:
        cwd = Path.cwd()
        candidates = [cwd / "README.md", cwd / "readme.md"]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                path = candidate
                print(f"INFO: missing path argument, defaulting to {path}")
                break
        if path is None:
            print("ERROR: missing path argument")
            return
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return
    if path.is_dir():
        print(f"ERROR: path is a directory: {path}")
        return

    data = path.read_bytes()
    text = _decode_content(data, args.encoding)
    lines = text.splitlines()
    start = max(0, args.offset or 0)
    end = len(lines) if args.limit <= 0 else min(len(lines), start + args.limit)
    output = "\n".join(lines[start:end])
    print(output)


if __name__ == "__main__":
    main()
