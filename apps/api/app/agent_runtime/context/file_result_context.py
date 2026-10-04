from __future__ import annotations

import logging
from typing import Any

from app.agent_runtime.context.resource_effects import append_resource_effect_context


logger = logging.getLogger(__name__)

LAST_FILE_OPERATION_RESULT_KEY = "last_file_operation_result"


def promote_filesystem_result_context(metadata: dict[str, Any] | None, result_payload: Any) -> dict[str, Any]:
    """Promote a verified filesystem result into next-turn file context.

    `pending_operation` remembers an unfinished action. This helper handles the
    opposite case: a finished action that produced a new file path, such as a
    copied or renamed file. Without this promotion, follow-ups like "刚才复制的文件"
    only exist in chat text and are easy to lose after unrelated turns.
    """

    next_metadata = dict(metadata or {})
    result = result_payload if isinstance(result_payload, dict) else {}
    if not result or result.get("ok") is False:
        return next_metadata

    trace = result.get("filesystem_trace") if isinstance(result.get("filesystem_trace"), dict) else {}
    postcheck = trace.get("postcheck") if isinstance(trace.get("postcheck"), dict) else {}
    if postcheck.get("completed") is not True:
        return next_metadata

    operation = str(result.get("operation") or trace.get("operation") or "").strip()
    arguments = result.get("arguments") if isinstance(result.get("arguments"), dict) else {}
    source_path = _first_text(postcheck.get("source_path"), arguments.get("src"), arguments.get("path"), postcheck.get("path"))
    target_path = _first_text(postcheck.get("target_path"), arguments.get("dst"))
    focus_path, focus_kind = _focus_for_operation(operation=operation, source_path=source_path, target_path=target_path)
    if not focus_path:
        return next_metadata

    next_metadata = append_resource_effect_context(
        next_metadata,
        _resource_effect_for_filesystem_result(
            operation=operation,
            source_path=source_path,
            target_path=target_path,
            focus_path=focus_path,
            focus_kind=focus_kind,
        ),
    )
    next_metadata["active_file"] = _active_file_metadata(focus_path, operation=operation, focus_kind=focus_kind)
    next_metadata[LAST_FILE_OPERATION_RESULT_KEY] = {
        "source": "filesystem_result_context_v1",
        "operation": operation,
        "source_path": source_path,
        "target_path": target_path,
        "focus_path": focus_path,
        "focus_kind": focus_kind,
        "completed": True,
    }
    next_metadata["recent_file_paths"] = _dedupe(
        [*_string_list(next_metadata.get("recent_file_paths")), source_path, target_path, focus_path]
    )
    logger.info(
        "Promoted filesystem result into file context: operation=%s focus_path=%s source_path=%s target_path=%s",
        operation,
        focus_path,
        source_path,
        target_path,
    )
    return next_metadata


def _resource_effect_for_filesystem_result(
    *,
    operation: str,
    source_path: str,
    target_path: str,
    focus_path: str,
    focus_kind: str,
) -> dict[str, Any]:
    return {
        "source": "filesystem_result_context_v1",
        "resource_type": "file",
        "action": _resource_action_for_filesystem_operation(operation),
        "operation": operation,
        "path": focus_path,
        "source_path": source_path,
        "target_path": target_path,
        "focus_path": focus_path,
        "focus_kind": focus_kind,
        "display_name": _path_filename(focus_path),
        "aliases": _aliases_for_filesystem_operation(operation),
        "completed": True,
    }


def _resource_action_for_filesystem_operation(operation: str) -> str:
    if operation == "copy_file":
        return "created"
    if operation == "rename_file":
        return "renamed"
    if operation in {"read_file", "path_exists"}:
        return "observed"
    return "updated"


def _aliases_for_filesystem_operation(operation: str) -> list[str]:
    if operation == "copy_file":
        return ["刚才复制的文件", "复制出来的文件", "新复制的文件"]
    if operation == "rename_file":
        return ["刚才改名的文件", "改名后的文件", "当前文件"]
    if operation == "read_file":
        return ["刚才读的文件", "这个文件", "当前文件"]
    if operation == "path_exists":
        return ["刚才确认的文件", "这个文件", "当前文件"]
    return ["当前文件"]


def _focus_for_operation(*, operation: str, source_path: str, target_path: str) -> tuple[str, str]:
    if operation == "copy_file" and target_path:
        return target_path, "file"
    if operation == "rename_file" and target_path:
        return target_path, "filename"
    if operation in {"read_file", "path_exists"} and source_path:
        return source_path, "content" if operation == "read_file" else "file"
    return "", "file"


def _active_file_metadata(path: str, *, operation: str, focus_kind: str) -> dict[str, Any]:
    directory, filename, stem, extension = _path_parts(path)
    return {
        "path": path,
        "directory": directory,
        "filename": filename,
        "stem": stem,
        "extension": extension,
        "last_action": operation or "file_reference",
        "last_focus": focus_kind or "file",
        "source": "filesystem_result",
    }


def _path_parts(path: str) -> tuple[str, str, str, str]:
    normalized = str(path or "").strip()
    index = max(normalized.rfind("/"), normalized.rfind("\\"))
    directory = normalized[:index] if index >= 0 else ""
    filename = normalized[index + 1 :] if index >= 0 else normalized
    if "." in filename:
        stem, extension = filename.rsplit(".", 1)
        return directory, filename, stem, f".{extension}" if extension else ""
    return directory, filename, filename, ""


def _path_filename(path: str) -> str:
    normalized = str(path or "").strip()
    index = max(normalized.rfind("/"), normalized.rfind("\\"))
    return normalized[index + 1 :] if index >= 0 else normalized


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


__all__ = ["LAST_FILE_OPERATION_RESULT_KEY", "promote_filesystem_result_context"]
