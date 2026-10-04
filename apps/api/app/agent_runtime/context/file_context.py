from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any

from app.agent_runtime.context.file_operation_policy import (
    extract_destination_directory_paths,
    generate_rename_destination_from_operation_intent,
    resolve_copy_destination_from_operation_intent,
)
from app.agent_runtime.context.filename_intent import is_placeholder_filename
from app.agent_runtime.context.resource_effects import latest_resource_effect


logger = logging.getLogger(__name__)

LOCAL_FILE_REFERENCE_RE = re.compile(
    r"[A-Za-z]:[\\/][^\r\n`\"<>]*?\.(?:tex|md|txt|pdf|docx|json|csv|yaml|yml)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ActiveFileContext:
    path: str
    directory: str
    filename: str
    stem: str
    extension: str
    last_action: str
    last_focus: str
    source: str

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "directory": self.directory,
            "filename": self.filename,
            "stem": self.stem,
            "extension": self.extension,
            "last_action": self.last_action,
            "last_focus": self.last_focus,
            "source": self.source,
        }


def build_file_context_hints(
    *,
    user_message: str,
    recent_user_context: str = "",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Carry factual file state without classifying prose."""

    context = context or {}
    current_paths = extract_local_file_references(user_message)
    recent_paths = extract_local_file_references(recent_user_context)
    context_paths = _string_list(context.get("recent_file_paths"))
    all_paths = _dedupe([*context_paths, *recent_paths, *current_paths])

    current_directories = extract_destination_directory_paths(user_message)
    recent_directories = extract_destination_directory_paths(recent_user_context)
    context_directories = _string_list(context.get("recent_directory_paths"))
    all_directories = _dedupe([*context_directories, *recent_directories, *current_directories])

    active_file = _active_file_from_context(context) or _active_file_from_resource_context(context)
    if current_paths:
        active_file = _active_file_from_path(current_paths[-1], source="user_message")
    elif active_file is None and recent_paths:
        active_file = _active_file_from_path(recent_paths[-1], source="recent_user_context")
    elif active_file is None and context_paths:
        active_file = _active_file_from_path(context_paths[-1], source="context_recent_file_paths")

    if active_file is not None:
        all_paths = _dedupe([*all_paths, active_file.path])

    hints: dict[str, Any] = {}
    if all_paths:
        hints["recent_file_paths"] = all_paths
    if all_directories:
        hints["recent_directory_paths"] = all_directories

    active_directory = _active_directory_from_context(context)
    if current_directories:
        active_directory = {"path": current_directories[-1], "source": "user_message"}
    elif active_directory is None and recent_directories:
        active_directory = {"path": recent_directories[-1], "source": "recent_user_context"}
    elif active_directory is None and context_directories:
        active_directory = {"path": context_directories[-1], "source": "context_recent_directory_paths"}
    if active_directory is not None:
        hints["active_directory"] = active_directory
    if active_file is not None:
        hints["active_file"] = active_file.to_metadata_dict()

    structured_operation = str(context.get("filesystem_operation") or "").strip()
    operation_intent = context.get("operation_intent")
    if structured_operation:
        hints["filesystem_operation"] = structured_operation
    if isinstance(operation_intent, dict) and operation_intent:
        hints["operation_intent"] = dict(operation_intent)

    logger.info(
        "Built factual file context: active_path=%s structured_operation=%s",
        (hints.get("active_file") or {}).get("path"),
        structured_operation or None,
    )
    return hints


def complete_move_file_arguments(
    *,
    tool_input: dict[str, Any],
    user_message: str,
    recent_user_context: str = "",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill only from explicit tool input or structured model intent."""

    context = context or {}
    hints = build_file_context_hints(
        user_message=user_message,
        recent_user_context=recent_user_context,
        context=context,
    )
    pending_known = _pending_known_args(context)
    active_file = hints.get("active_file") if isinstance(hints.get("active_file"), dict) else {}

    source_path = str(tool_input.get("src") or tool_input.get("path") or "").strip()
    if not source_path:
        source_path = str(pending_known.get("src") or pending_known.get("path") or "").strip()
    if not source_path:
        source_path = str(active_file.get("path") or "").strip()
    if not source_path:
        source_path = _best_available_path(
            user_message=user_message,
            recent_user_context=recent_user_context,
            context={**context, **hints},
        )

    operation_intent = _structured_operation_intent(tool_input, context, pending_known)
    target_path = generate_rename_destination_from_operation_intent(operation_intent, source_path)
    target_origin = "model_operation_intent" if target_path else ""
    raw_target = str(tool_input.get("dst") or "").strip()
    if not raw_target:
        raw_target = str(pending_known.get("dst") or "").strip()
        if raw_target:
            target_origin = "pending_structured_input"
    if raw_target and not is_placeholder_filename(raw_target) and not target_path:
        target_path = raw_target
        target_origin = target_origin or "explicit_tool_input"
    if is_placeholder_filename(target_path):
        target_path = ""

    completed = dict(tool_input)
    if is_placeholder_filename(str(completed.get("dst") or "")):
        completed.pop("dst", None)
    if source_path and not str(completed.get("src") or "").strip():
        completed["src"] = source_path
    if target_path:
        completed["dst"] = target_path
    completed.setdefault("overwrite", False)

    if target_path:
        logger.info(
            "Filesystem rename completion: src=%s dst=%s target_origin=%s",
            source_path,
            target_path,
            target_origin,
        )
    else:
        logger.warning(
            "Filesystem rename completion incomplete: missing=dst src=%s context_keys=%s",
            source_path,
            sorted(context.keys()),
        )
    return completed


def complete_copy_file_arguments(
    *,
    tool_input: dict[str, Any],
    user_message: str,
    recent_user_context: str = "",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill copy arguments without deriving a target from conversational prose."""

    context = context or {}
    hints = build_file_context_hints(
        user_message=user_message,
        recent_user_context=recent_user_context,
        context=context,
    )
    pending_known = _pending_known_args(context)
    active_file = hints.get("active_file") if isinstance(hints.get("active_file"), dict) else {}

    source_path = str(tool_input.get("src") or tool_input.get("path") or "").strip()
    if not source_path:
        source_path = str(pending_known.get("src") or pending_known.get("path") or "").strip()
    if not source_path:
        source_path = str(active_file.get("path") or "").strip()
    if not source_path:
        source_path = _best_available_path(
            user_message=user_message,
            recent_user_context=recent_user_context,
            context={**context, **hints},
        )

    operation_intent = _structured_operation_intent(tool_input, context, pending_known)
    target_path = resolve_copy_destination_from_operation_intent(operation_intent, source_path)
    target_origin = "model_operation_intent" if target_path else ""
    if not target_path:
        target_path = str(tool_input.get("dst") or "").strip()
        target_origin = "explicit_tool_input" if target_path else ""
    if not target_path:
        target_path = str(pending_known.get("dst") or "").strip()
        target_origin = "pending_structured_input" if target_path else ""
    if is_placeholder_filename(target_path):
        target_path = ""

    completed = dict(tool_input)
    if source_path and not str(completed.get("src") or "").strip():
        completed["src"] = source_path
    if target_path:
        completed["dst"] = target_path
    completed.setdefault("overwrite", False)

    if source_path and target_path:
        logger.info(
            "Filesystem copy completion: src=%s dst=%s target_origin=%s",
            source_path,
            target_path,
            target_origin,
        )
    else:
        missing = [field for field, value in (("src", source_path), ("dst", target_path)) if not value]
        logger.warning(
            "Filesystem copy completion incomplete: missing=%s target_origin=%s context_keys=%s",
            missing,
            target_origin,
            sorted(context.keys()),
        )
    return completed


def extract_local_file_references(text: str) -> list[str]:
    if not text:
        return []
    return _dedupe(match.group(0).strip() for match in LOCAL_FILE_REFERENCE_RE.finditer(text))


def _structured_operation_intent(
    tool_input: dict[str, Any],
    context: dict[str, Any],
    pending_known: dict[str, Any],
) -> dict[str, Any]:
    for candidate in (
        tool_input.get("operation_intent"),
        context.get("operation_intent"),
        pending_known.get("operation_intent"),
    ):
        if isinstance(candidate, dict) and candidate:
            return dict(candidate)
    return {}


def _pending_known_args(context: dict[str, Any]) -> dict[str, Any]:
    pending = context.get("pending_operation")
    if not isinstance(pending, dict):
        return {}
    known_args = pending.get("known_args")
    return dict(known_args) if isinstance(known_args, dict) else {}


def _best_available_path(*, user_message: str, recent_user_context: str, context: dict[str, Any]) -> str:
    direct_paths = extract_local_file_references(user_message)
    if direct_paths:
        return direct_paths[-1]
    context_paths = context.get("recent_file_paths")
    if isinstance(context_paths, list):
        for path in reversed(context_paths):
            cleaned = str(path or "").strip()
            if cleaned:
                return cleaned
    recent_paths = extract_local_file_references(recent_user_context)
    if recent_paths:
        return recent_paths[-1]
    active_file = context.get("active_file")
    return str(active_file.get("path") or "").strip() if isinstance(active_file, dict) else ""


def _active_file_from_context(context: dict[str, Any]) -> ActiveFileContext | None:
    active = context.get("active_file")
    if not isinstance(active, dict):
        return None
    path = str(active.get("path") or "").strip()
    if not path:
        return None
    parts = _path_parts(path)
    return ActiveFileContext(
        path=path,
        directory=parts["directory"],
        filename=parts["filename"],
        stem=parts["stem"],
        extension=parts["extension"],
        last_action=str(active.get("last_action") or "file_reference"),
        last_focus=str(active.get("last_focus") or "file"),
        source=str(active.get("source") or "context_metadata"),
    )


def _active_file_from_resource_context(context: dict[str, Any]) -> ActiveFileContext | None:
    resource = context.get("active_resource") if isinstance(context.get("active_resource"), dict) else {}
    if not resource or str(resource.get("resource_type") or "") != "file":
        resource = latest_resource_effect(context, resource_type="file")
    path = str(resource.get("focus_path") or resource.get("target_path") or resource.get("path") or "").strip()
    if not path:
        return None
    parts = _path_parts(path)
    return ActiveFileContext(
        path=path,
        directory=parts["directory"],
        filename=parts["filename"],
        stem=parts["stem"],
        extension=parts["extension"],
        last_action=str(resource.get("operation") or resource.get("action") or "file_reference"),
        last_focus=str(resource.get("focus_kind") or "file"),
        source="resource_effect",
    )


def _active_directory_from_context(context: dict[str, Any]) -> dict[str, str] | None:
    active_directory = context.get("active_directory")
    if not isinstance(active_directory, dict):
        return None
    path = str(active_directory.get("path") or "").strip().rstrip("/\\")
    if not path:
        return None
    return {
        "path": path,
        "source": str(active_directory.get("source") or "context_metadata"),
    }


def _active_file_from_path(path: str, *, source: str) -> ActiveFileContext:
    parts = _path_parts(path)
    return ActiveFileContext(
        path=path,
        directory=parts["directory"],
        filename=parts["filename"],
        stem=parts["stem"],
        extension=parts["extension"],
        last_action="file_reference",
        last_focus="file",
        source=source,
    )


def _path_parts(path: str) -> dict[str, str]:
    normalized = str(path or "").strip()
    separator_index = max(normalized.rfind("/"), normalized.rfind("\\"))
    directory = normalized[:separator_index] if separator_index >= 0 else ""
    filename = normalized[separator_index + 1 :] if separator_index >= 0 else normalized
    if "." in filename:
        stem, extension = filename.rsplit(".", 1)
        extension = f".{extension}" if extension else ""
    else:
        stem, extension = filename, ""
    return {"directory": directory, "filename": filename, "stem": stem, "extension": extension}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


__all__ = [
    "ActiveFileContext",
    "build_file_context_hints",
    "complete_copy_file_arguments",
    "complete_move_file_arguments",
    "extract_local_file_references",
]
