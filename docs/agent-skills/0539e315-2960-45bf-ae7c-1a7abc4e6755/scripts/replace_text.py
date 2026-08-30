from __future__ import annotations

import argparse

from cli_utils import get_flag_value, resolve_path


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
    parser = argparse.ArgumentParser(description="Replace exact text in a local file.")
    parser.add_argument("path", nargs="?", help="Absolute file path.")
    parser.add_argument("--path", dest="path_flag", help="Absolute file path.")
    parser.add_argument("--old-text", dest="old_text", help="Exact text to replace.")
    parser.add_argument("--new-text", dest="new_text", help="Replacement text.")
    parser.add_argument("--encoding", default="utf-8", help="Text encoding.")
    parser.add_argument("--count", type=int, default=0, help="Max replacements; 0 means all occurrences.")
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
        print(f"ERROR: file not found: {path}")
        return
    if path.is_dir():
        print(f"ERROR: path is a directory: {path}")
        return

    old_text = args.old_text or get_flag_value(unknown, ["--old-text", "--old", "--from"])
    new_text = args.new_text or get_flag_value(unknown, ["--new-text", "--new", "--to"])
    if old_text is None or old_text == "":
        print("ERROR: old_text is required")
        return
    if new_text is None:
        print("ERROR: new_text is required")
        return

    data = path.read_bytes()
    text = _decode_content(data, args.encoding)
    replace_count = text.count(old_text) if args.count <= 0 else min(text.count(old_text), args.count)
    if replace_count <= 0:
        print(f"ERROR: old_text not found: {old_text}")
        return

    updated = text.replace(old_text, new_text, args.count if args.count > 0 else -1)
    output_encoding = args.encoding if args.encoding.lower() not in {"auto", "detect"} else "utf-8"
    path.write_bytes(updated.encode(output_encoding))
    print(f"REPLACED: {replace_count} occurrence(s) in {path}")


if __name__ == "__main__":
    main()
