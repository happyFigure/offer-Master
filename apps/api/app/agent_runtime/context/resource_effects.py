from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)

RESOURCE_EFFECTS_KEY = "resource_effects"
ACTIVE_RESOURCE_KEY = "active_resource"
ARTIFACT_CONTEXT_KEY = "artifact_context"


def append_resource_effect_context(
    metadata: dict[str, Any] | None,
    effect: dict[str, Any] | None,
    *,
    max_effects: int = 20,
) -> dict[str, Any]:
    """Append a generic tool effect and promote it as the active resource.

    This is the my-agent-style bridge between tool execution and later natural
    references. Tools should report what resource they created, changed or read;
    follow-up turns can then resolve "刚才那个" from structured state instead of
    re-parsing chat text.
    """

    next_metadata = dict(metadata or {})
    normalized = _normalize_effect(effect)
    if not normalized:
        return next_metadata

    existing = [item for item in next_metadata.get(RESOURCE_EFFECTS_KEY, []) if isinstance(item, dict)]
    next_metadata[RESOURCE_EFFECTS_KEY] = [*existing, normalized][-max_effects:]
    next_metadata[ACTIVE_RESOURCE_KEY] = normalized
    if normalized.get("resource_type") == "file":
        next_metadata[ARTIFACT_CONTEXT_KEY] = _file_artifact_context(next_metadata, normalized)

    logger.info(
        "Recorded resource effect: resource_type=%s action=%s operation=%s focus=%s",
        normalized.get("resource_type"),
        normalized.get("action"),
        normalized.get("operation"),
        normalized.get("focus_path") or normalized.get("target_path") or normalized.get("path"),
    )
    return next_metadata


def promote_declared_resource_effects_context(metadata: dict[str, Any] | None, result_payload: Any) -> dict[str, Any]:
    """Promote resource effects explicitly declared by any tool result.

    This is the generic contract for future tools: they do not need custom
    runtime code if they return `resource_effects` in the standardized shape.
    Domain-specific adapters, such as filesystem postcheck handling, can still
    translate legacy results into the same contract.
    """

    next_metadata = dict(metadata or {})
    result = result_payload if isinstance(result_payload, dict) else {}
    if not result or result.get("ok") is False:
        return next_metadata

    effects = result.get(RESOURCE_EFFECTS_KEY)
    if isinstance(effects, dict):
        effects = [effects]
    if not isinstance(effects, list):
        return next_metadata

    for effect in effects:
        if not isinstance(effect, dict):
            continue
        next_metadata = append_resource_effect_context(next_metadata, effect)
    return next_metadata


def promote_contract_resource_effects_context(
    metadata: dict[str, Any] | None,
    result_payload: Any,
    *,
    tool_input: dict[str, Any] | None = None,
    semantic_profile: Any | None = None,
) -> dict[str, Any]:
    """Promote resource effects generated from a tool semantic contract.

    A tool can keep its natural output shape and declare how that output maps to
    runtime resources in `semantic_profile.result_contract`. This keeps the
    closed loop extensible: adding a new artifact-producing tool should change
    its registration contract, not the agent loop.
    """

    next_metadata = dict(metadata or {})
    result = result_payload if isinstance(result_payload, dict) else {}
    if not result or result.get("ok") is False:
        return next_metadata

    contract = getattr(semantic_profile, "result_contract", None)
    if not isinstance(contract, dict):
        return next_metadata
    templates = contract.get("resource_effect_templates")
    if isinstance(templates, dict):
        templates = [templates]
    if not isinstance(templates, list):
        return next_metadata

    bindings = {"input": dict(tool_input or {}), "result": result}
    for template in templates:
        if not isinstance(template, dict):
            continue
        effect = _render_effect_template(template, bindings=bindings)
        next_metadata = append_resource_effect_context(next_metadata, effect)
    return next_metadata


def latest_resource_effect(
    metadata: dict[str, Any] | None,
    *,
    resource_type: str = "",
    operation: str = "",
    action: str = "",
) -> dict[str, Any]:
    """Return the newest matching effect from runtime metadata."""

    if not isinstance(metadata, dict):
        return {}
    effects = metadata.get(RESOURCE_EFFECTS_KEY)
    if not isinstance(effects, list):
        return {}
    for effect in reversed(effects):
        if not isinstance(effect, dict):
            continue
        if resource_type and str(effect.get("resource_type") or "") != resource_type:
            continue
        if operation and str(effect.get("operation") or "") != operation:
            continue
        if action and str(effect.get("action") or "") != action:
            continue
        return dict(effect)
    return {}


def _render_effect_template(template: dict[str, Any], *, bindings: dict[str, Any]) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for key, value in template.items():
        if isinstance(value, str):
            rendered[key] = _resolve_template_value(value, bindings=bindings)
        elif isinstance(value, list):
            rendered[key] = [_resolve_template_value(item, bindings=bindings) if isinstance(item, str) else item for item in value]
        else:
            rendered[key] = value
    return rendered


def _resolve_template_value(value: str, *, bindings: dict[str, Any]) -> str:
    text = str(value or "").strip()
    if not text.startswith("$"):
        return text
    return str(_field_path_value(bindings, text[1:]) or "").strip()


def _field_path_value(payload: dict[str, Any], field_path: str) -> Any:
    current: Any = payload
    for key in field_path.split("."):
        if not key:
            return None
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _normalize_effect(effect: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(effect, dict):
        return {}
    resource_type = str(effect.get("resource_type") or "").strip()
    action = str(effect.get("action") or "").strip()
    operation = str(effect.get("operation") or "").strip()
    if not resource_type or not action:
        return {}
    return {
        "source": str(effect.get("source") or "resource_effects_v1"),
        "resource_type": resource_type,
        "action": action,
        "operation": operation,
        "path": str(effect.get("path") or "").strip(),
        "source_path": str(effect.get("source_path") or "").strip(),
        "target_path": str(effect.get("target_path") or "").strip(),
        "focus_path": str(effect.get("focus_path") or effect.get("target_path") or effect.get("path") or "").strip(),
        "focus_kind": str(effect.get("focus_kind") or "resource").strip(),
        "display_name": str(effect.get("display_name") or "").strip(),
        "aliases": _string_list(effect.get("aliases")),
        "completed": bool(effect.get("completed") if "completed" in effect else True),
    }


def _file_artifact_context(metadata: dict[str, Any], effect: dict[str, Any]) -> dict[str, Any]:
    current = metadata.get(ARTIFACT_CONTEXT_KEY) if isinstance(metadata.get(ARTIFACT_CONTEXT_KEY), dict) else {}
    created = [item for item in current.get("created_artifacts", []) if isinstance(item, dict)] if isinstance(current, dict) else []
    if effect.get("action") == "created":
        created = [*created, effect][-20:]
    return {
        **dict(current or {}),
        "active_artifact": effect,
        "created_artifacts": created,
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


__all__ = [
    "ACTIVE_RESOURCE_KEY",
    "ARTIFACT_CONTEXT_KEY",
    "RESOURCE_EFFECTS_KEY",
    "append_resource_effect_context",
    "latest_resource_effect",
    "promote_contract_resource_effects_context",
    "promote_declared_resource_effects_context",
]
