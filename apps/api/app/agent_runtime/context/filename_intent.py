from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable


_PLACEHOLDER_STEMS = frozenset(
    {
        "你起的名字",
        "你自己起的名字",
        "你推荐的名字",
        "你取的名字",
        "合适的名字",
        "合适的名称",
        "刚才那个名字",
        "刚才的名字",
        "上面那个名字",
        "文件名",
        "文件名称",
        "名字",
        "名称",
        "你起名字",
        "你自己起名字",
        "一下",
        "改一下",
        "修改一下",
        "换一下",
        "处理一下",
        "弄一下",
        "搞一下",
        "合适",
        "随便",
    }
)


def is_placeholder_filename(value: str) -> bool:
    """Reject conversational placeholders before they reach a filesystem script."""

    stem = _path_filename(str(value or "")).strip()
    if not stem:
        return True
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    compact = re.sub(r"[\s_\-.]+", "", stem)
    normalized_placeholders = {
        re.sub(r"[\s_\-.]+", "", item) for item in _PLACEHOLDER_STEMS
    }
    return compact in normalized_placeholders


def normalize_model_filename(value: str, *, source_path: str = "") -> str:
    """Turn a model-proposed display name into one safe filename.

    The model chooses the semantic name. This function only strips Markdown,
    rejects path injection, normalizes Windows-invalid punctuation, and keeps
    the source extension when the model omitted one.
    """

    filename = str(value or "").strip()
    if not filename or len(filename) > 160 or is_placeholder_filename(filename):
        return ""
    if any(separator in filename for separator in ("/", "\\")) or ".." in filename:
        return ""

    filename = re.sub(r"^[#>*_\`\s]+|[#>*_\`\s]+$", "", filename)
    filename = re.sub(r"[《》「」『』【】]", "", filename)
    filename = filename.replace("×", "_").replace("·", "_")
    filename = re.sub(r"[：:*?\"<>|]", "_", filename)
    filename = re.sub(r"[\r\n]+", " ", filename)
    filename = re.sub(r"\s+", "_", filename)
    filename = re.sub(r"_+", "_", filename).strip(" ._-")
    if not filename or is_placeholder_filename(filename):
        return ""

    source_extension = _path_extension(source_path)
    if source_extension and not _path_extension(filename):
        filename = f"{filename}{source_extension}"
    return filename


def resolve_rename_destination_from_name_intent(
    operation_intent: dict[str, Any] | None,
    source_path: str,
    *,
    path_exists: Callable[[str], bool] | None = None,
) -> str:
    """Resolve a structured model name while keeping path rules in runtime."""

    if not isinstance(operation_intent, dict) or not source_path:
        return ""
    destination = operation_intent.get("destination")
    name_intent = operation_intent.get("name_intent")
    if not isinstance(destination, dict) or not isinstance(name_intent, dict):
        return ""

    kind = str(destination.get("kind") or "").strip().lower()
    destination_path = str(destination.get("path") or "").strip().rstrip("/\\")
    if kind not in {"file", "directory"}:
        return ""

    candidate = (
        name_intent.get("filename_stem")
        or name_intent.get("filename")
        or name_intent.get("display_name")
    )
    filename = normalize_model_filename(str(candidate or ""), source_path=source_path)
    if not filename:
        return ""

    source_directory = _path_directory(source_path)
    if kind == "directory":
        directory = destination_path or source_directory
    else:
        directory = destination_path if destination_path and not is_placeholder_filename(destination_path) else source_directory
        if directory and _path_filename(directory) == filename:
            directory = _path_directory(directory)
    if not directory:
        return filename

    separator = "\\" if "\\" in directory and "/" not in directory else "/"
    clean_directory = directory.rstrip("/\\")
    target = f"{clean_directory}{separator}{filename}"
    exists = path_exists or (lambda candidate_path: Path(candidate_path).exists())
    if exists(target):
        return ""
    return target


def _path_filename(path: str) -> str:
    normalized = str(path or "").strip()
    return normalized[max(normalized.rfind("/"), normalized.rfind("\\")) + 1 :]


def _path_directory(path: str) -> str:
    normalized = str(path or "").strip()
    index = max(normalized.rfind("/"), normalized.rfind("\\"))
    return normalized[:index] if index >= 0 else ""


def _path_extension(path: str) -> str:
    filename = _path_filename(path)
    if "." not in filename:
        return ""
    extension = filename.rsplit(".", 1)[1]
    return f".{extension}" if extension else ""


__all__ = [
    "is_placeholder_filename",
    "normalize_model_filename",
    "resolve_rename_destination_from_name_intent",
]
