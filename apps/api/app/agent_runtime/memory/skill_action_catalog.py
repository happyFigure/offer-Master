from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ACTION_SCHEMA_FILE_KEYS = ("actionSchemasFile", "action_schemas_file")
ACTION_SCHEMA_KEYS = ("actionSchemas", "action_schemas")
MODEL_ALLOW_KEYS = ("modelAllow", "model_allow", "model_allowed_actions")
MODEL_DENY_KEYS = ("modelDeny", "model_deny", "model_denied_actions")
EXECUTE_ALLOW_KEYS = ("executeAllow", "execute_allow", "execute_allowed_actions")
EXECUTE_DENY_KEYS = ("executeDeny", "execute_deny", "execute_denied_actions")


def build_skill_action_metadata(
    *,
    package_root: Path,
    frontmatter: dict[str, Any],
    resources: dict[str, list[str]],
) -> dict[str, Any]:
    """Normalize my-agents/OpenClaw action metadata into one runtime shape.

    Skill authors write human docs in SKILL.md, but action execution needs a
    machine-readable map: action name -> script path -> input schema. Keeping
    that normalization here lets list/read/call tools share the same source of
    truth instead of each tool reparsing frontmatter differently.
    """
    openclaw_metadata = parse_openclaw_metadata(frontmatter.get("metadata"))
    action_policy = _action_policy(openclaw_metadata)
    action_schemas_file = _first_string(action_policy, ACTION_SCHEMA_FILE_KEYS)
    action_schemas = _load_action_schemas(
        package_root=package_root,
        inline_schemas=_first_mapping(action_policy, ACTION_SCHEMA_KEYS),
        schema_file=action_schemas_file,
    )
    model_allow = _string_list(_first_value(action_policy, MODEL_ALLOW_KEYS))
    model_deny = _string_list(_first_value(action_policy, MODEL_DENY_KEYS))
    execute_allow = _string_list(_first_value(action_policy, EXECUTE_ALLOW_KEYS))
    execute_deny = _string_list(_first_value(action_policy, EXECUTE_DENY_KEYS))
    script_map = _script_action_map(resources)

    candidate_actions = _visible_action_names(
        script_actions=set(script_map),
        schema_actions=set(action_schemas),
        model_allow=model_allow,
        model_deny=model_deny,
    )
    action_map = {action: script_map[action] for action in candidate_actions if action in script_map}
    action_details = [
        _action_detail(
            action=action,
            schema=action_schemas.get(action),
            script_path=action_map.get(action),
        )
        for action in candidate_actions
    ]

    return {
        "openclaw_metadata": openclaw_metadata,
        "actions": candidate_actions,
        "action_map": action_map,
        "action_details": action_details,
        "action_schemas_file": action_schemas_file,
        "action_schemas": action_schemas,
        "model_allowed_actions": model_allow,
        "model_denied_actions": model_deny,
        "execute_allowed_actions": execute_allow,
        "execute_denied_actions": execute_deny,
    }


def parse_openclaw_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        metadata = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        metadata = parsed if isinstance(parsed, dict) else {}
    else:
        metadata = {}

    openclaw = metadata.get("openclaw") if isinstance(metadata, dict) else None
    return openclaw if isinstance(openclaw, dict) else metadata


def action_details_from_metadata(metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    metadata = metadata or {}
    details = metadata.get("action_details")
    if not isinstance(details, list):
        return []
    return [dict(item) for item in details if isinstance(item, dict) and str(item.get("action") or "").strip()]


def resolve_skill_action(
    metadata: dict[str, Any] | None,
    requested_action: str | None,
) -> dict[str, Any] | None:
    details = action_details_from_metadata(metadata)
    if requested_action:
        normalized = normalize_action_name(requested_action)
        for detail in details:
            if normalize_action_name(str(detail.get("action") or "")) == normalized:
                return detail
        return None
    executable = [detail for detail in details if str(detail.get("script_path") or "").strip()]
    return executable[0] if len(executable) == 1 else None


def validate_skill_action_arguments(action_detail: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    schema = action_detail.get("input_schema") if isinstance(action_detail.get("input_schema"), dict) else {}
    required = schema.get("required") if isinstance(schema, dict) else []
    properties = schema.get("properties") if isinstance(schema, dict) else {}
    missing = [str(name) for name in required if str(name) not in arguments]
    extra: list[str] = []
    if schema.get("additionalProperties") is False and isinstance(properties, dict):
        extra = sorted(set(arguments) - set(properties))

    problems: list[str] = []
    if missing:
        problems.append(f"missing required arguments: {', '.join(missing)}")
    if extra:
        problems.append(f"unsupported arguments: {', '.join(extra)}")
    return "; ".join(problems) or None


def resolve_skill_script_path(skill_file: Path, action_detail: dict[str, Any]) -> Path | None:
    relative = str(action_detail.get("script_path") or "").strip()
    if not relative:
        return None
    root = skill_file.parent.resolve()
    script_path = (root / relative).resolve()
    # Only scripts inside the imported Skill package are executable. This is the
    # same boundary my-agents enforces when it maps actions to scripts.
    try:
        script_path.relative_to(root)
    except ValueError:
        return None
    return script_path if script_path.is_file() else None


def skill_action_command_arguments(arguments: dict[str, Any]) -> list[str]:
    command: list[str] = []
    for key, value in arguments.items():
        if value is None:
            continue
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False)
        else:
            rendered = str(value)
        command.extend([flag, rendered])
    return command


def normalize_action_name(value: str) -> str:
    return str(value or "").strip().replace("-", "_")


def _action_policy(openclaw_metadata: dict[str, Any]) -> dict[str, Any]:
    actions = openclaw_metadata.get("actions") if isinstance(openclaw_metadata, dict) else None
    return actions if isinstance(actions, dict) else openclaw_metadata


def _load_action_schemas(
    *,
    package_root: Path,
    inline_schemas: dict[str, Any],
    schema_file: str,
) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    if schema_file:
        schemas.update(_load_action_schemas_file(package_root, schema_file))
    for action, schema in inline_schemas.items():
        if isinstance(schema, dict):
            schemas[normalize_action_name(action)] = dict(schema)
    return schemas


def _load_action_schemas_file(package_root: Path, schema_file: str) -> dict[str, dict[str, Any]]:
    raw_path = Path(schema_file)
    if raw_path.is_absolute() or ".." in raw_path.parts:
        return {}
    target = (package_root / raw_path).resolve()
    try:
        target.relative_to(package_root.resolve())
    except ValueError:
        return {}
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {normalize_action_name(action): dict(schema) for action, schema in payload.items() if isinstance(schema, dict)}


def _script_action_map(resources: dict[str, list[str]]) -> dict[str, str]:
    action_map: dict[str, str] = {}
    for resource in resources.get("scripts") or []:
        path = Path(resource)
        if path.suffix != ".py" or path.stem.startswith("_"):
            continue
        action_map[normalize_action_name(path.stem)] = path.as_posix()
    return dict(sorted(action_map.items()))


def _visible_action_names(
    *,
    script_actions: set[str],
    schema_actions: set[str],
    model_allow: list[str],
    model_deny: list[str],
) -> list[str]:
    available = script_actions | schema_actions
    if model_allow:
        candidates = [normalize_action_name(action) for action in model_allow]
    else:
        candidates = sorted(available)
    denied = {normalize_action_name(action) for action in model_deny}
    return [action for action in candidates if action in available and action not in denied]


def _action_detail(action: str, schema: dict[str, Any] | None, script_path: str | None) -> dict[str, Any]:
    schema = schema or {}
    input_schema = _input_schema(schema)
    properties = input_schema.get("properties") if isinstance(input_schema.get("properties"), dict) else {}
    source_parts = [part for part in ("script" if script_path else "", "schema" if schema else "") if part]
    return {
        "action": action,
        "description": str(schema.get("description") or f"Run Skill action {action}."),
        "script_path": script_path,
        "input_schema": input_schema,
        "required": list(input_schema.get("required") or []),
        "parameter_names": list(properties),
        "source": "+".join(source_parts) or "metadata",
    }


def _input_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
        return {
            "type": "object",
            "properties": dict(schema.get("properties") or {}),
            "required": list(schema.get("required") or []),
            "additionalProperties": bool(schema.get("additionalProperties", True)),
        }
    return {
        "type": "object",
        "properties": dict(schema.get("properties") or {}) if isinstance(schema.get("properties"), dict) else {},
        "required": list(schema.get("required") or []),
        "additionalProperties": bool(schema.get("additionalProperties", True)),
    }


def _first_value(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _first_string(mapping: dict[str, Any], keys: tuple[str, ...]) -> str:
    value = _first_value(mapping, keys)
    return str(value or "").strip()


def _first_mapping(mapping: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    value = _first_value(mapping, keys)
    return dict(value) if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [normalize_action_name(str(item)) for item in value if str(item).strip()]
    return [normalize_action_name(item) for item in str(value).split(",") if item.strip()]


__all__ = [
    "action_details_from_metadata",
    "build_skill_action_metadata",
    "normalize_action_name",
    "parse_openclaw_metadata",
    "resolve_skill_action",
    "resolve_skill_script_path",
    "skill_action_command_arguments",
    "validate_skill_action_arguments",
]
