from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from cli_utils import (
    collect_positional,
    get_first_positional,
    get_flag_value,
    normalize_path,
    resolve_text,
)

_DEFAULT_OUTPUT_ENV = "MY_AGENTS_DEFAULT_FILE_OUTPUT_DIR"


def _default_output_path() -> Path:
    override = os.environ.get(_DEFAULT_OUTPUT_ENV, "").strip()
    root = Path(override).expanduser() if override else Path(tempfile.gettempdir()) / "my-agents"
    return root.resolve() / "output.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write text to a file.")
    parser.add_argument("path", nargs="?", help="Absolute file path.")
    parser.add_argument("--path", dest="path_flag", help="Absolute file path.")
    parser.add_argument("text_pos", nargs="?", help="Positional text content.")
    parser.add_argument("--text", dest="text_value", help="Text content to write.")
    parser.add_argument(
        "--content",
        help="Alias for --text (kept for compatibility with model output).",
    )
    parser.add_argument(
        "--data",
        help="Alias for --text (kept for compatibility with model output).",
    )
    parser.add_argument("--encoding", default="utf-8", help="Text encoding.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwrite if file exists.")
    args, unknown = parser.parse_known_args()

    path_value = None
    path_from_positional = False
    if args.path_flag:
        path_value = args.path_flag
    else:
        path_flag_from_unknown = get_flag_value(
            unknown,
            ["--path", "-p", "--file-path", "--filepath", "--target"],
        )
        if path_flag_from_unknown:
            path_value = path_flag_from_unknown
        elif args.path:
            path_value = args.path
            path_from_positional = True
        else:
            path_value = get_first_positional(unknown)
            path_from_positional = path_value is not None

    path = normalize_path(path_value)
    if path is None:
        path = _default_output_path()
        print(f"INFO: missing path argument, defaulting to {path}")

    text = resolve_text(
        args.text_value or args.content or args.data,
        ["--text", "--content", "--data"],
        unknown,
    )
    if text is None:
        positional_tokens: list[str] = []
        if args.path is not None:
            positional_tokens.append(args.path)
        if args.text_pos is not None:
            positional_tokens.append(args.text_pos)
        positional_tokens.extend(collect_positional(unknown))
        if path_from_positional and positional_tokens:
            positional_tokens = positional_tokens[1:]
        text = " ".join(positional_tokens) if positional_tokens else None
    if text is None:
        parser.error("text content is required (use --text or a positional value)")
    if path.exists() and not args.overwrite:
        print(f"ERROR: file exists (use --overwrite to replace): {path}")
        return
    if path.exists() and path.is_dir():
        print(f"ERROR: path is a directory: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=args.encoding)
    print(f"WROTE: {path}")


if __name__ == "__main__":
    main()
