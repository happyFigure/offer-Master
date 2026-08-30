from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence


def _clean_arg(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return value.strip().strip('"').strip("'")


def normalize_path(value: Optional[str]) -> Optional[Path]:
    cleaned = _clean_arg(value)
    if not cleaned:
        return None
    path = Path(cleaned).expanduser()
    try:
        return path.resolve(strict=False)
    except RuntimeError:
        return path


def get_flag_value(args: Sequence[str], names: Iterable[str]) -> Optional[str]:
    name_set = set(names)
    for idx, item in enumerate(args):
        if item in name_set and idx + 1 < len(args):
            return args[idx + 1]
        for name in name_set:
            prefix = f"{name}="
            if item.startswith(prefix):
                return item[len(prefix) :]
    return None


def get_first_positional(args: Sequence[str]) -> Optional[str]:
    for item in args:
        if item.startswith("-"):
            continue
        return item
    return None


def collect_positional(args: Sequence[str]) -> list[str]:
    return [item for item in args if not item.startswith("-")]


def get_nth_positional(args: Sequence[str], index: int) -> Optional[str]:
    positional = collect_positional(args)
    if 0 <= index < len(positional):
        return positional[index]
    return None


def resolve_path(
    primary: Optional[str],
    fallback_flags: Iterable[str],
    unknown: Sequence[str],
) -> Optional[Path]:
    candidate = primary or get_flag_value(unknown, fallback_flags) or get_first_positional(unknown)
    return normalize_path(candidate)


def resolve_text(
    primary: Optional[str],
    fallback_flags: Iterable[str],
    unknown: Sequence[str],
) -> Optional[str]:
    candidate = primary or get_flag_value(unknown, fallback_flags)
    if candidate:
        return _clean_arg(candidate)
    positional = collect_positional(unknown)
    if positional:
        return " ".join(positional)
    return None
