from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable

from app.agent_runtime.context.filename_intent import (
    is_placeholder_filename,
    normalize_model_filename,
    resolve_rename_destination_from_name_intent,
)


_DIRECTORY_REFERENCE_MARKERS = (
    "目录",
    "文件夹",
    "folder",
    "directory",
    "复制到",
    "拷贝到",
    "备份到",
    "另存到",
    "放到",
    "放进",
    "保存到",
    "目标位置",
)
_ABSOLUTE_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\r\n`\"<>，。；,;！？!?（）()]+")
_KNOWN_FILE_EXTENSIONS = frozenset({"tex", "md", "txt", "pdf", "docx", "json", "csv", "yaml", "yml"})


def resolve_copy_destination_from_operation_intent(
    operation_intent: dict[str, Any] | None,
    source_path: str,
    *,
    path_exists: Callable[[str], bool] | None = None,
) -> str:
    """Resolve only a model-submitted copy destination.

    The runtime may normalize a concrete file target or a model-provided
    ``name_intent`` inside a declared directory. It deliberately does not
    invent ``-副本``/numeric names when the model omitted the semantic name.
    """

    if not isinstance(operation_intent, dict):
        return ""

    destination = operation_intent.get("destination")
    if not isinstance(destination, dict):
        return ""

    kind = str(destination.get("kind") or "").strip().lower()
    destination_path = str(destination.get("path") or "").strip().rstrip("/\\")
    if not destination_path:
        return ""

    # The model owns the semantic choice, but the runtime still rejects a
    # candidate whose declared kind conflicts with an existing local path.
    # This prevents a directory reference from silently becoming a filename
    # (or the reverse) when the filesystem script resolves the destination.
    exists = path_exists or (lambda candidate: Path(candidate).exists())
    if exists(destination_path):
        try:
            destination_is_directory = Path(destination_path).is_dir()
        except (OSError, ValueError):
            return ""
        if (kind == "directory") != destination_is_directory:
            return ""

    if kind == "file":
        return "" if is_placeholder_filename(destination_path) else destination_path
    if kind != "directory":
        return ""

    requested_policy = str(operation_intent.get("name_policy") or "").strip().lower()
    if requested_policy not in {"content_based", "content_based_name", "name_from_content"}:
        return ""

    # The model owns the semantic filename. Runtime only normalizes the
    # submitted name, joins it to the declared destination boundary, and
    # rejects conflicts; it never invents a fallback such as "-副本".
    return _resolve_content_based_copy_destination(
        operation_intent,
        source_path,
        destination_kind=kind,
        destination_path=destination_path,
        path_exists=path_exists,
    )


def _resolve_content_based_copy_destination(
    operation_intent: dict[str, Any],
    source_path: str,
    *,
    destination_kind: str,
    destination_path: str,
    path_exists: Callable[[str], bool] | None,
) -> str:
    name_intent = operation_intent.get("name_intent")
    if not isinstance(name_intent, dict):
        return ""
    candidate = name_intent.get("filename_stem") or name_intent.get("filename") or name_intent.get("display_name")
    filename = normalize_model_filename(str(candidate or ""), source_path=source_path)
    if not filename:
        return ""

    if destination_kind == "directory":
        directory = destination_path or _split_path(source_path)[0]
    elif destination_kind == "file":
        directory = _split_path(destination_path)[0] or _split_path(source_path)[0]
    else:
        return ""
    if not directory:
        return filename

    separator = _separator_for(directory, source_path)
    target = f"{directory.rstrip('/\\')}{separator}{filename}"
    exists = path_exists or (lambda candidate_path: Path(candidate_path).exists())
    return "" if exists(target) else target


def generate_rename_destination_from_operation_intent(
    operation_intent: dict[str, Any] | None,
    source_path: str,
    *,
    path_exists: Callable[[str], bool] | None = None,
) -> str:
    """Normalize a model-selected rename target without inventing a filename."""

    if not isinstance(operation_intent, dict):
        return ""
    structured_name_target = resolve_rename_destination_from_name_intent(
        operation_intent,
        source_path,
        path_exists=path_exists,
    )
    if structured_name_target:
        return structured_name_target

    destination = operation_intent.get("destination")
    if not isinstance(destination, dict):
        return ""
    kind = str(destination.get("kind") or "").strip().lower()
    destination_path = str(destination.get("path") or "").strip().rstrip("/\\")
    if kind != "file" or not destination_path or is_placeholder_filename(destination_path):
        return ""

    exists = path_exists or (lambda candidate: Path(candidate).exists())
    if exists(destination_path):
        try:
            if Path(destination_path).is_dir():
                return ""
        except (OSError, ValueError):
            return ""
    return destination_path


def extract_destination_directory_paths(text: str) -> list[str]:
    """Extract explicit local destination directories from natural-language text."""

    raw_text = str(text or "")
    if not raw_text:
        return []
    results: list[str] = []
    for match in _ABSOLUTE_PATH_RE.finditer(raw_text):
        candidate = match.group(0).strip().strip(" ：:，,。！？?；;").rstrip("\\/")
        if not candidate or _looks_like_known_file_path(candidate):
            continue
        nearby_text = raw_text[max(0, match.start() - 40) : min(len(raw_text), match.end() + 20)]
        if candidate != raw_text.strip() and not any(marker.lower() in nearby_text.lower() for marker in _DIRECTORY_REFERENCE_MARKERS):
            continue
        if candidate not in results:
            results.append(candidate)
    return results


def _separator_for(primary_path: str, fallback_path: str = "") -> str:
    path = primary_path or fallback_path
    return "\\" if "\\" in path and "/" not in path else "/"


def _split_path(path: str) -> tuple[str, str, str]:
    # Keep the separator style from the original user-visible path. pathlib on
    # Windows may normalize C:/foo into C:\foo, which breaks conflict checks that
    # compare literal strings from prior context.
    slash_index = max(path.rfind("/"), path.rfind("\\"))
    if slash_index < 0:
        filename = path
        directory = ""
    else:
        directory = path[:slash_index]
        filename = path[slash_index + 1 :]
    if "." in filename:
        stem, extension = filename.rsplit(".", 1)
        return directory, stem, f".{extension}"
    return directory, filename, ""


def _looks_like_known_file_path(path: str) -> bool:
    filename = path[max(path.rfind("/"), path.rfind("\\")) + 1 :]
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[-1].lower()
    return extension in _KNOWN_FILE_EXTENSIONS


__all__ = [
    "extract_destination_directory_paths",
    "generate_rename_destination_from_operation_intent",
    "resolve_copy_destination_from_operation_intent",
]
