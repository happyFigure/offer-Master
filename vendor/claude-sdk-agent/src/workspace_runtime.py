from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

logger = logging.getLogger(__name__)

WORKSPACE_RUNTIME_SCHEMA = "claude-code.workspace-runtime/v1"
_MAX_ITEMS_PER_RESOURCE = 200
_MAX_SCAN_DEPTH = 8
_MAX_FRONTMATTER_BYTES = 128 * 1024
_MAX_JSON_BYTES = 2 * 1024 * 1024
_DYNAMIC_COMMAND_PATTERN = re.compile(r"^/[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_PERMISSION_EFFECTS = ("allow", "ask", "deny")
_PERMISSION_RULE_PRIORITY = ("deny", "ask", "allow")
_PERMISSION_EVALUATION_ORDER = (
    "pre_tool_use_hooks",
    "deny_rules",
    "ask_rules",
    "permission_mode",
    "allow_rules",
    "can_use_tool",
)


@dataclass(frozen=True, slots=True)
class WorkspaceRoot:
    path: Path
    role: str
    native_scope: str


def inspect_workspace_runtime(
    *,
    cwd: Path,
    add_dirs: Sequence[Path] = (),
    source: str = "agent_default",
    configured: bool = False,
    setting_sources: Sequence[str] | None = None,
    strict_mcp_config: bool = False,
    permission_profile: str = "",
    permission_mode: str = "",
    allowed_tools: Sequence[str] | None = None,
    disallowed_tools: Sequence[str] | None = None,
    permission_full_bypass: bool = False,
    permission_runtime_key: str = "",
    permission_revision: int = 0,
    agent_config_dir: Path | None = None,
    skill_mount_root: Path | None = None,
    skill_platform_catalog: str = "include",
    workflow_mount_root: Path | None = None,
    agent_mcp_names: Sequence[str] = (),
    additional_directories_claude_md: bool = False,
) -> dict[str, Any]:
    """Describe native Claude Code resources without executing workspace code.

    The primary cwd is the only root treated as a project configuration root.
    Additional directories are reported as access roots. This mirrors Claude
    Code's distinction between ``cwd`` and ``--add-dir`` and avoids silently
    importing settings, hooks, or MCP servers from every mounted directory.
    """

    primary = _normalize_directory(cwd)
    if primary is None:
        raise ValueError(f"workspace directory does not exist: {cwd}")
    roots = [WorkspaceRoot(primary, "primary", "project")]
    additional_roots = []
    for item in add_dirs:
        path = _normalize_directory(item)
        if path is None or path == primary or any(root.path == path for root in roots):
            continue
        root = WorkspaceRoot(path, "additional", "access")
        roots.append(root)
        additional_roots.append(root)

    sources = _normalize_sources(setting_sources)
    project_enabled = sources is None or "project" in sources
    local_enabled = sources is None or "local" in sources
    user_enabled = sources is None or "user" in sources

    primary_scan = _scan_root(roots[0], project_enabled=project_enabled, local_enabled=local_enabled)
    additional_scans = [
        _scan_root(root, project_enabled=False, local_enabled=False)
        for root in additional_roots
    ]
    agent_scans = _scan_agent_roots(
        agent_config_dir=agent_config_dir,
        skill_mount_root=skill_mount_root,
        workflow_mount_root=workflow_mount_root,
        user_enabled=user_enabled,
    )

    resources = _merge_resource_views(primary_scan, additional_scans, agent_scans)
    platform_catalog = _normalize_platform_catalog(skill_platform_catalog)
    _apply_platform_skill_policy(
        resources,
        skill_mount_root=skill_mount_root,
        platform_catalog=platform_catalog,
    )
    if additional_directories_claude_md:
        _activate_additional_memory(additional_scans, resources)
    mcp = _build_mcp_view(
        primary_scan,
        agent_mcp_names=agent_mcp_names,
        strict_mcp_config=strict_mcp_config,
        project_enabled=project_enabled,
    )
    permission = _build_permission_view(
        resources,
        profile=permission_profile,
        mode=permission_mode,
        allowed_tools=allowed_tools,
        disallowed_tools=disallowed_tools,
        full_bypass=permission_full_bypass,
        runtime_key=permission_runtime_key,
        revision=permission_revision,
    )
    hooks = _build_hooks_view(resources)
    commands = _build_command_catalog(resources)
    conflicts = _find_conflicts(commands=commands, mcp=mcp, permission=permission)
    fingerprint = _fingerprint(resources, roots, agent_scans)

    additional_policy = {
        "mode": "access_only",
        "claudeMd": {
            "enabled": bool(additional_directories_claude_md),
            "reason": "explicit_environment_policy"
            if additional_directories_claude_md
            else "add_dir_does_not_load_project_memory_by_default",
        },
        "settings": "not_loaded",
        "mcp": "not_loaded",
        "hooks": "not_loaded",
    }

    return {
        "schemaVersion": WORKSPACE_RUNTIME_SCHEMA,
        "status": "ready",
        "workspace": {
            "cwd": str(primary),
            "addDirs": [str(root.path) for root in additional_roots],
            "source": str(source or "agent_default"),
            "configured": bool(configured),
            "roots": [
                {
                    "path": str(root.path),
                    "role": root.role,
                    "nativeScope": root.native_scope,
                }
                for root in roots
            ],
        },
        "agentPolicy": {
            "skills": {
                "platformCatalog": platform_catalog,
                "platformMountEnabled": platform_catalog == "include",
                "runtimeKey": str(permission_runtime_key or ""),
            },
            "settingSources": list(sources) if sources is not None else None,
            "userSettingsEnabled": user_enabled,
            "projectSettingsEnabled": project_enabled,
            "localSettingsEnabled": local_enabled,
            "permission": permission,
            "mcp": {
                "strictConfig": bool(strict_mcp_config),
                "serverNames": list(mcp["agentServerNames"]),
            },
            "additionalDirectories": additional_policy,
        },
        "resources": resources,
        "commands": commands,
        "effectiveRuntime": {
            "resolution": "claude_code_native",
            "cwd": str(primary),
            "addDirs": [str(root.path) for root in additional_roots],
            "skills": {
                "platformCatalog": platform_catalog,
                "platformMountEnabled": platform_catalog == "include",
            },
            "settingSources": list(sources) if sources is not None else ["user", "project", "local"],
            "permission": permission,
            "mcp": mcp,
            "hooks": hooks,
            "resourceCounts": {
                name: int(value.get("activeCount") or 0)
                for name, value in resources.items()
                if isinstance(value, Mapping)
            },
        },
        "conflicts": conflicts,
        "fingerprint": fingerprint,
        "observed": {
            "available": False,
            "reason": "not_connected",
        },
    }


def merge_observed_runtime(
    runtime: Mapping[str, Any],
    *,
    server_info: Mapping[str, Any] | None = None,
    mcp_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach SDK-observed data while preserving the static declaration view."""

    result = _deep_copy_mapping(runtime)
    observed: dict[str, Any] = {"available": True}
    if isinstance(server_info, Mapping):
        commands = server_info.get("commands")
        if isinstance(commands, list):
            observed["commands"] = [_normalize_observed_command(item) for item in commands]
        for key in ("outputStyles", "models", "account", "sessionId"):
            if key in server_info and key != "account":
                observed[key] = _safe_observed_value(server_info[key])
    if isinstance(mcp_status, Mapping):
        observed["mcp"] = _normalize_observed_mcp_status(mcp_status)
    result["observed"] = observed
    effective = result.get("effectiveRuntime")
    if isinstance(effective, dict) and isinstance(observed.get("mcp"), Mapping):
        mcp_view = effective.get("mcp")
        if isinstance(mcp_view, dict):
            statuses = observed["mcp"].get("mcpServers")
            if isinstance(statuses, list):
                mcp_view["serverStatuses"] = statuses
    return result


def _normalize_directory(value: Any) -> Path | None:
    if value is None:
        return None
    try:
        path = Path(value).expanduser().resolve()
    except (OSError, TypeError, ValueError):
        return None
    if not path.exists() or not path.is_dir():
        return None
    return path


def _normalize_sources(value: Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    allowed = {"user", "project", "local"}
    result: list[str] = []
    for item in value:
        name = str(item or "").strip().lower()
        if name in allowed and name not in result:
            result.append(name)
    return result


def _scan_root(root: WorkspaceRoot, *, project_enabled: bool, local_enabled: bool) -> dict[str, Any]:
    path = root.path
    claude_dir = path / ".claude"
    result: dict[str, Any] = {
        "root": str(path),
        "role": root.role,
        "scope": root.native_scope,
        "memory": [],
        "settings": [],
        "mcp": [],
        "hooks": [],
        "permissions": [],
        "skills": [],
        "commands": [],
        "outputStyles": [],
        "agents": [],
        "workflows": [],
    }

    memory_paths = (path / "CLAUDE.md", claude_dir / "CLAUDE.md")
    for memory_path in memory_paths:
        if memory_path.is_file():
            result["memory"].append(
                _item(memory_path, path, "memory", scope="project", active=project_enabled)
            )
    agents_path = path / "AGENTS.md"
    if agents_path.is_file():
        imported = any(_claude_imports_agents(memory_path) for memory_path in memory_paths if memory_path.is_file())
        if imported:
            result["memory"].append(
                _item(
                    agents_path,
                    path,
                    "memory_import",
                    scope="project",
                    active=project_enabled,
                    status="active" if project_enabled else "detected_only",
                    reason="imported_by_claude_md",
                )
            )

    project_settings = claude_dir / "settings.json"
    local_settings = claude_dir / "settings.local.json"
    for settings_path, scope, active in (
        (project_settings, "project", project_enabled),
        (local_settings, "local", local_enabled),
    ):
        if settings_path.is_file():
            settings_item = _item(settings_path, path, "settings", scope=scope, active=active)
            settings_item.update(_summarize_settings(settings_path))
            result["settings"].append(settings_item)
            if settings_item.get("hookEvents"):
                result["hooks"].append(
                    _item(
                        settings_path,
                        path,
                        "hooks",
                        scope=scope,
                        active=active,
                        status="declared",
                        events=settings_item["hookEvents"],
                    )
                )
            if settings_item.get("hasPermissionConfig"):
                result["permissions"].append(
                    _item(
                        settings_path,
                        path,
                        "permissions",
                        scope=scope,
                        active=active,
                        status="declared",
                        **_permission_item_fields(settings_item),
                    )
                )

    mcp_path = path / ".mcp.json"
    if mcp_path.is_file():
        mcp_item = _item(
            mcp_path,
            path,
            "mcp",
            scope="project",
            active=project_enabled,
        )
        mcp_item.update(_summarize_mcp(mcp_path))
        result["mcp"].append(mcp_item)

    _append_markdown_items(result["skills"], claude_dir / "skills", path, "skill", project_enabled, scope="project", require_name="SKILL.md")
    _append_markdown_items(result["commands"], claude_dir / "commands", path, "command", project_enabled, scope="project", suffixes={".md"})
    _append_markdown_items(result["outputStyles"], claude_dir / "output-styles", path, "output_style", project_enabled, scope="project", suffixes={".md"})
    _append_markdown_items(result["agents"], claude_dir / "agents", path, "agent", project_enabled, scope="project", suffixes={".md"})
    _append_markdown_items(result["workflows"], claude_dir / "workflows", path, "workflow", project_enabled, scope="project", suffixes={".js"})

    rules_dir = claude_dir / "rules"
    for file_path in _bounded_files(rules_dir, suffixes={".md"}):
        result["memory"].append(
            _item(file_path, path, "rule", scope="project", active=project_enabled)
        )
    memory_dir = claude_dir / "agent-memory"
    for file_path in _bounded_files(memory_dir, names={"MEMORY.md"}):
        result["memory"].append(
            _item(file_path, path, "agent_memory", scope="project", active=project_enabled)
        )

    _tag_scan_items(result, root_role=root.role, root_path=path)
    return result


def _scan_agent_roots(
    *,
    agent_config_dir: Path | None,
    skill_mount_root: Path | None,
    workflow_mount_root: Path | None,
    user_enabled: bool,
) -> list[dict[str, Any]]:
    roots: list[Path] = []
    for value in (agent_config_dir, skill_mount_root, workflow_mount_root):
        path = _normalize_directory(value)
        if path is not None and path not in roots:
            roots.append(path)
    scans: list[dict[str, Any]] = []
    for path in roots:
        root = WorkspaceRoot(path, "agent", "agent")
        scan = _scan_agent_root(root, user_enabled=user_enabled)
        if any(
            scan.get(key)
            for key in (
                "memory",
                "settings",
                "mcp",
                "hooks",
                "permissions",
                "skills",
                "commands",
                "outputStyles",
                "agents",
                "workflows",
            )
        ):
            scans.append(scan)
    return scans


def _scan_agent_root(root: WorkspaceRoot, *, user_enabled: bool) -> dict[str, Any]:
    path = root.path
    result: dict[str, Any] = {
        "root": str(path),
        "role": "agent",
        "scope": "agent",
        "memory": [],
        "settings": [],
        "mcp": [],
        "hooks": [],
        "permissions": [],
        "skills": [],
        "commands": [],
        "outputStyles": [],
        "agents": [],
        "workflows": [],
    }
    scan_root = path
    is_user_config_root = path.name == ".claude"
    agent_resource_active = user_enabled if is_user_config_root or path.parent.name == ".claude" else True
    if is_user_config_root:
        scan_root = path.parent
    claude_dir = scan_root / ".claude"
    if path.name == ".claude":
        claude_dir = path
        user_memory = claude_dir / "CLAUDE.md"
        if user_memory.is_file():
            result["memory"].append(
                _item(user_memory, scan_root, "memory", scope="user", active=user_enabled)
            )
        user_settings = claude_dir / "settings.json"
        if user_settings.is_file():
            settings_item = _item(user_settings, scan_root, "settings", scope="user", active=user_enabled)
            settings_item.update(_summarize_settings(user_settings))
            result["settings"].append(settings_item)
            if settings_item.get("hookEvents"):
                result["hooks"].append(
                    _item(
                        user_settings,
                        scan_root,
                        "hooks",
                        scope="user",
                        active=user_enabled,
                        status="declared",
                        events=settings_item["hookEvents"],
                    )
                )
            if settings_item.get("hasPermissionConfig"):
                result["permissions"].append(
                    _item(
                        user_settings,
                        scan_root,
                        "permissions",
                        scope="user",
                        active=user_enabled,
                        status="declared",
                        **_permission_item_fields(settings_item),
                    )
                )
    _append_markdown_items(result["skills"], claude_dir / "skills", scan_root, "skill", agent_resource_active, scope="agent", require_name="SKILL.md")
    _append_markdown_items(result["commands"], claude_dir / "commands", scan_root, "command", agent_resource_active, scope="agent", suffixes={".md"})
    _append_markdown_items(result["outputStyles"], claude_dir / "output-styles", scan_root, "output_style", agent_resource_active, scope="agent", suffixes={".md"})
    _append_markdown_items(result["agents"], claude_dir / "agents", scan_root, "agent", agent_resource_active, scope="agent", suffixes={".md"})
    _append_markdown_items(result["workflows"], claude_dir / "workflows", scan_root, "workflow", agent_resource_active, scope="agent", suffixes={".js"})
    if path.name == "workflows":
        _append_files_as_items(result["workflows"], path, scan_root, "workflow", agent_resource_active, scope="agent", suffixes={".js"})
    _tag_scan_items(result, root_role="agent", root_path=path)
    return result


def _tag_scan_items(scan: Mapping[str, Any], *, root_role: str, root_path: Path) -> None:
    for value in scan.values():
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                item["rootRole"] = root_role
                item["root"] = str(root_path)


def _merge_resource_views(
    primary: Mapping[str, Any],
    additional: Sequence[Mapping[str, Any]],
    agent: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    names = ("memory", "settings", "mcp", "hooks", "permissions", "skills", "commands", "outputStyles", "agents", "workflows")
    resources: dict[str, Any] = {}
    for name in names:
        primary_items = [dict(item) for item in primary.get(name, []) if isinstance(item, Mapping)]
        additional_items = [
            dict(item)
            for scan in additional
            for item in scan.get(name, [])
            if isinstance(item, Mapping)
        ]
        agent_items = [
            dict(item)
            for scan in agent
            for item in scan.get(name, [])
            if isinstance(item, Mapping)
        ]
        all_items = _unique_items(primary_items + additional_items + agent_items)
        resources[name] = {
            "detectedCount": len(all_items),
            "activeCount": sum(1 for item in all_items if item.get("active")),
            "items": all_items,
            "truncated": bool(
                primary.get(f"{name}Truncated")
                or any(scan.get(f"{name}Truncated") for scan in additional)
                or any(scan.get(f"{name}Truncated") for scan in agent)
            ),
        }
    return resources


def _normalize_platform_catalog(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in {"include", "exclude"} else "include"


def _apply_platform_skill_policy(
    resources: Mapping[str, Any],
    *,
    skill_mount_root: Path | None,
    platform_catalog: str,
) -> None:
    if platform_catalog != "exclude" or skill_mount_root is None:
        return
    skills = resources.get("skills")
    if not isinstance(skills, dict):
        return
    try:
        mount_key = os.path.normcase(str(Path(skill_mount_root).expanduser().resolve()))
    except (OSError, TypeError, ValueError):
        return
    excluded_count = 0
    for item in skills.get("items", []):
        if not isinstance(item, dict):
            continue
        try:
            root_key = os.path.normcase(str(Path(str(item.get("root") or "")).expanduser().resolve()))
        except (OSError, TypeError, ValueError):
            continue
        if root_key != mount_key:
            continue
        item["active"] = False
        item["status"] = "excluded"
        item["reason"] = "platform_catalog_excluded"
        item["reasonCodes"] = ["platform_catalog_excluded"]
        excluded_count += 1
    skills["activeCount"] = sum(
        1 for item in skills.get("items", [])
        if isinstance(item, Mapping) and item.get("active")
    )
    skills["excludedCount"] = excluded_count


def _build_command_catalog(resources: Mapping[str, Any]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for kind in ("skills", "commands", "workflows"):
        resource = resources.get(kind)
        if not isinstance(resource, Mapping):
            continue
        for item in resource.get("items", []):
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            resource_active = bool(item.get("active"))
            user_invocable = _metadata_bool(item.get("userInvocable"), default=True)
            support_file = kind == "commands" and _is_command_support_file(item)
            active = resource_active and user_invocable and not support_file
            entry_kind = {"skills": "skill", "commands": "command", "workflows": "workflow"}[kind]
            entry = {
                "id": f"{kind}:{item.get('path')}",
                "name": name,
                "command": f"/{name}",
                "kind": entry_kind,
                "description": str(item.get("description") or "").strip(),
                "argumentHint": str(item.get("argumentHint") or "").strip(),
                "path": str(item.get("path") or ""),
                "source": str(item.get("source") or "workspace"),
                "resourceActive": resource_active,
                "userInvocable": user_invocable,
                "active": active,
            }
            if entry_kind == "workflow":
                entry.update(
                    {
                        "requiresConfirmation": False,
                        "confirmationMode": "none",
                        "launchMode": "sdk_immediate",
                        "permissionBehavior": "runtime_permission_mode",
                    }
                )
            command_token = f"/{name}"
            entry["invokable"] = active and bool(_DYNAMIC_COMMAND_PATTERN.fullmatch(command_token))
            if entry["invokable"]:
                entry["invoke"] = {
                    "source": "claude-code",
                    "kind": entry["kind"],
                    "commandId": name.lower(),
                    "command": command_token,
                    "displayName": name,
                    "acceptsArguments": True,
                }
                if entry_kind == "workflow":
                    entry["invoke"].update(
                        {
                            "requiresConfirmation": False,
                            "confirmationMode": "none",
                            "launchMode": "sdk_immediate",
                        }
                    )
            elif not resource_active:
                entry["inactiveReason"] = "resource_inactive"
            elif support_file:
                entry["inactiveReason"] = "support_file"
            elif not user_invocable:
                entry["inactiveReason"] = "not_user_invocable"
            else:
                entry["inactiveReason"] = "unsupported_command_name"
            items.append(entry)
    by_name: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_name.setdefault(str(item["name"]).lower(), []).append(item)
    for entries in by_name.values():
        if len(entries) <= 1:
            continue
        # Claude Code treats a skill with the same name as the command-facing
        # definition. Keep every declaration visible, but mark the winner.
        priority = {"skill": 0, "command": 1, "workflow": 2}
        active_entries = [item for item in entries if item.get("active")]
        winner = min(active_entries or entries, key=lambda item: priority.get(str(item.get("kind") or ""), 99))
        for item in entries:
            item["conflict"] = True
            item["selected"] = item is winner
    items.sort(key=lambda item: (not bool(item.get("active")), str(item.get("name") or "").lower(), str(item.get("kind") or "")))
    return {
        "detectedCount": len(items),
        "activeCount": sum(1 for item in items if item.get("active")),
        "items": items,
    }


def _metadata_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return default


def _is_command_support_file(item: Mapping[str, Any]) -> bool:
    relative_path = str(item.get("relativePath") or "").strip()
    if not relative_path:
        return False
    path = Path(relative_path)
    if path.name.lower() in {"readme.md", "readme.markdown"}:
        return True
    parts = path.parts
    try:
        commands_index = parts.index("commands")
    except ValueError:
        return False
    return any(part.startswith("_") for part in parts[commands_index + 1 : -1])


def _build_mcp_view(
    primary: Mapping[str, Any],
    *,
    agent_mcp_names: Sequence[str],
    strict_mcp_config: bool,
    project_enabled: bool,
) -> dict[str, Any]:
    workspace_names = [
        str(name).strip()
        for item in primary.get("mcp", [])
        if isinstance(item, Mapping)
        for name in item.get("serverNames", [])
        if str(name).strip()
    ]
    workspace_names = list(dict.fromkeys(workspace_names))
    agent_names = list(dict.fromkeys(str(name).strip() for name in agent_mcp_names if str(name).strip()))
    collisions = sorted(set(workspace_names) & set(agent_names))
    active_workspace = bool(project_enabled and not strict_mcp_config)
    active_settings = [
        item
        for item in primary.get("settings", [])
        if isinstance(item, Mapping) and item.get("active")
    ]
    enable_all = any(item.get("enableAllProjectMcpServers") for item in active_settings)
    explicitly_enabled = list(dict.fromkeys(
        str(name).strip()
        for item in active_settings
        for name in item.get("enabledMcpServers", [])
        if str(name).strip()
    ))
    selection_configured = bool(
        explicitly_enabled
        or any(item.get("hasEnableAllProjectMcpServers") for item in active_settings)
    )
    selected_workspace_names = list(workspace_names)
    if active_workspace and selection_configured and not enable_all:
        selected_workspace_names = [name for name in workspace_names if name in explicitly_enabled]
    if not active_workspace:
        selected_workspace_names = []
    return {
        "strictConfig": bool(strict_mcp_config),
        "workspaceConfigDetected": bool(workspace_names),
        "workspaceServerNames": workspace_names,
        "workspaceServersActive": active_workspace,
        "workspaceServerSelection": (
            "all" if enable_all else "explicit" if selection_configured else "native_default"
        ),
        "workspaceEnabledServerNames": selected_workspace_names,
        "workspaceDisabledServerNames": [name for name in workspace_names if name not in selected_workspace_names],
        "workspaceInactiveReason": (
            "strict_mcp_config" if strict_mcp_config else "project_settings_source_disabled" if not project_enabled else ""
        ),
        "agentServerNames": agent_names,
        "expectedServerNames": list(dict.fromkeys(agent_names + selected_workspace_names)),
        "nameCollisions": collisions,
        "serverStatuses": [],
    }


def _build_permission_view(
    resources: Mapping[str, Any],
    *,
    profile: str,
    mode: str,
    allowed_tools: Sequence[str] | None,
    disallowed_tools: Sequence[str] | None,
    full_bypass: bool,
    runtime_key: str,
    revision: int,
) -> dict[str, Any]:
    resource = resources.get("permissions") if isinstance(resources.get("permissions"), Mapping) else {}
    items = [item for item in resource.get("items", []) if isinstance(item, Mapping)]
    detected_rules = _permission_counts()
    settings_rules = _permission_counts()
    workspace_rules = _permission_counts()
    user_rules = _permission_counts()
    rules: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    settings_default_modes: list[dict[str, Any]] = []
    disable_bypass_sources: list[dict[str, str]] = []

    for item in items:
        counts = _normalized_permission_counts(item.get("ruleCounts"))
        _merge_permission_counts(detected_rules, counts)
        if not item.get("active"):
            continue
        _merge_permission_counts(settings_rules, counts)
        scope = str(item.get("scope") or "").strip()
        path = str(item.get("path") or "")
        if scope in {"project", "local"}:
            _merge_permission_counts(workspace_rules, counts)
        if scope in {"user", "agent"}:
            _merge_permission_counts(user_rules, counts)
        source = {
            "scope": scope,
            "path": path,
            "active": True,
        }
        sources.append(source)
        declarations = item.get("rules") if isinstance(item.get("rules"), Mapping) else {}
        for effect in _PERMISSION_EFFECTS:
            for rule in _string_list(declarations.get(effect)):
                rules.append(_permission_rule(effect, rule, scope=scope, path=path))
        default_mode = str(item.get("defaultMode") or "").strip()
        if default_mode:
            settings_default_modes.append({
                "permissionMode": default_mode,
                "scope": scope,
                "path": path,
                "selected": False,
                "ignoredReason": "agent_runtime_mode_precedence" if mode else "",
            })
        if item.get("disableBypassPermissionsMode"):
            disable_bypass_sources.append({"scope": scope, "path": path})

    agent_allowed = _string_list(allowed_tools)
    agent_denied = _string_list(disallowed_tools)
    for effect, values in (("allow", agent_allowed), ("deny", agent_denied)):
        for rule in values:
            rules.append(_permission_rule(effect, rule, scope="agent_runtime", path=""))
    rules.sort(
        key=lambda item: (
            _PERMISSION_RULE_PRIORITY.index(str(item["effect"])),
            str(item["scope"]),
            str(item["rule"]),
        )
    )
    effective_rules = _permission_counts()
    for item in rules:
        effective_rules[str(item["effect"])] += 1
    disable_bypass = bool(disable_bypass_sources)
    hooks_resource = resources.get("hooks") if isinstance(resources.get("hooks"), Mapping) else {}
    active_hooks = [
        item
        for item in hooks_resource.get("items", [])
        if isinstance(item, Mapping) and item.get("active")
    ]
    full_bypass_limitations: list[str] = []
    if settings_rules["deny"]:
        full_bypass_limitations.append("deny_rules")
    if settings_rules["ask"]:
        full_bypass_limitations.append("ask_rules")
    if disable_bypass:
        full_bypass_limitations.append("disable_bypass_permissions_mode")
    if active_hooks:
        full_bypass_limitations.append("workspace_hooks_may_block")
    return {
        "profile": str(profile or "").strip(),
        "permissionMode": str(mode or "").strip(),
        "permissionModeSource": "agent_runtime" if mode else "claude_code_settings",
        "runtimeKey": str(runtime_key or "").strip(),
        "revision": max(0, int(revision)),
        "agentAllowedTools": agent_allowed,
        "agentDisallowedTools": agent_denied,
        "workspaceRuleCounts": workspace_rules,
        "workspaceDetectedRuleCounts": detected_rules,
        "workspaceRuleSources": list(dict.fromkeys(
            str(item["scope"])
            for item in sources
            if item["scope"] in {"project", "local"}
        )),
        "workspaceRulesActive": bool(any(workspace_rules.values())),
        "userRuleCounts": user_rules,
        "settingsRuleCounts": settings_rules,
        "effectiveRuleCounts": effective_rules,
        "rules": rules,
        "ruleGroups": {
            effect: [item for item in rules if item["effect"] == effect]
            for effect in _PERMISSION_RULE_PRIORITY
        },
        "sources": sources,
        "settingsDefaultModes": settings_default_modes,
        "disableBypassPermissionsMode": disable_bypass,
        "disableBypassSources": disable_bypass_sources,
        "fullBypass": {
            "requested": bool(full_bypass),
            "nativePermissionMode": "bypassPermissions" if full_bypass else "",
            "enforcementStatus": (
                "workspace_policy_adapter_required"
                if full_bypass and full_bypass_limitations
                else "native_permission_mode"
                if full_bypass
                else "not_requested"
            ),
            "workspaceRulesIgnored": False,
            "workspaceHooksIgnored": False,
            "limitations": full_bypass_limitations if full_bypass else [],
        },
        "rulePriority": list(_PERMISSION_RULE_PRIORITY),
        "evaluationOrder": list(_PERMISSION_EVALUATION_ORDER),
        "precedence": "agent_mode_over_settings_default_mode;_native_rule_priority_preserved",
        "resolutionNote": "runtime_recalculation_is_read_only_and_does_not_modify_agent_or_workspace_configuration",
    }


def _permission_counts() -> dict[str, int]:
    return {effect: 0 for effect in _PERMISSION_EFFECTS}


def _normalized_permission_counts(value: Any) -> dict[str, int]:
    source = value if isinstance(value, Mapping) else {}
    result = _permission_counts()
    for effect in _PERMISSION_EFFECTS:
        try:
            result[effect] = max(0, int(source.get(effect) or 0))
        except (TypeError, ValueError):
            result[effect] = 0
    return result


def _merge_permission_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    for effect in _PERMISSION_EFFECTS:
        target[effect] += int(source.get(effect) or 0)


def _permission_rule(effect: str, rule: str, *, scope: str, path: str) -> dict[str, Any]:
    text = str(rule or "").strip()
    tool = text
    specifier = ""
    if "(" in text and text.endswith(")"):
        tool, specifier = text.split("(", 1)
        tool = tool.strip()
        specifier = specifier[:-1].strip()
    return {
        "effect": effect,
        "rule": text,
        "tool": tool,
        "specifier": specifier,
        "scope": scope,
        "path": path,
        "removesToolDefinition": effect == "deny" and not specifier,
        "preApprovalOnly": effect == "allow",
    }


def _build_hooks_view(resources: Mapping[str, Any]) -> dict[str, Any]:
    hooks_resource = resources.get("hooks") if isinstance(resources.get("hooks"), Mapping) else {}
    items = [item for item in hooks_resource.get("items", []) if isinstance(item, Mapping)]
    events = sorted({str(event) for item in items for event in item.get("events", []) if str(event).strip()})
    active = [item for item in items if item.get("active")]
    return {
        "detected": bool(items),
        "events": events,
        "declarationCount": len(items),
        "activeDeclarationCount": len(active),
        "active": bool(active),
        "source": [str(item.get("scope") or "") for item in active],
        "resolutionNote": "workspace_hooks_are_loaded_by_claude_code;_sdk_hooks_are_observability_callbacks",
    }


def _append_markdown_items(
    target: list[dict[str, Any]],
    directory: Path,
    root: Path,
    kind: str,
    active: bool,
    *,
    scope: str,
    require_name: str | None = None,
    suffixes: set[str] | None = None,
) -> None:
    if require_name:
        candidates = [item / require_name for item in _bounded_directories(directory)]
    else:
        candidates = _bounded_files(directory, suffixes=suffixes or {".md"})
    _append_files_as_items(target, candidates, root, kind, active, scope=scope, is_explicit_files=True)


def _append_files_as_items(
    target: list[dict[str, Any]],
    files_or_directory: Iterable[Path] | Path,
    root: Path,
    kind: str,
    active: bool,
    *,
    scope: str,
    suffixes: set[str] | None = None,
    is_explicit_files: bool = False,
) -> None:
    if isinstance(files_or_directory, Path):
        candidates = _bounded_files(files_or_directory, suffixes=suffixes)
    else:
        candidates = list(files_or_directory)
    for file_path in candidates[:_MAX_ITEMS_PER_RESOURCE]:
        if not file_path.is_file():
            continue
        metadata = _frontmatter_metadata(file_path) if file_path.suffix.lower() in {".md", ".markdown"} else {}
        name = str(metadata.get("name") or "").strip()
        if not name:
            if kind == "skill":
                name = file_path.parent.name
            else:
                name = file_path.stem
        item = _item(file_path, root, kind, scope=scope, active=active)
        item["name"] = name
        metadata_keys = {
            "description": "description",
            "argumentHint": "argumentHint",
            "argument_hint": "argumentHint",
            "argument-hint": "argumentHint",
            "userInvocable": "userInvocable",
            "user-invocable": "userInvocable",
            "disableModelInvocation": "disableModelInvocation",
            "disable-model-invocation": "disableModelInvocation",
        }
        for key, normalized_key in metadata_keys.items():
            if key in metadata:
                item[normalized_key] = metadata[key]
        target.append(item)


def _unique_items(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (str(item.get("kind") or ""), str(item.get("path") or ""))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _bounded_directories(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    try:
        entries = sorted((item for item in directory.iterdir() if item.is_dir()), key=lambda item: item.name.lower())
    except OSError:
        return []
    return entries[:_MAX_ITEMS_PER_RESOURCE]


def _bounded_files(
    directory: Path,
    *,
    suffixes: set[str] | None = None,
    names: set[str] | None = None,
) -> list[Path]:
    if not directory.is_dir():
        return []
    result: list[Path] = []
    suffixes = {str(item).lower() for item in (suffixes or set())}
    names = names or set()
    try:
        candidates = directory.rglob("*")
        for path in candidates:
            try:
                relative_depth = len(path.relative_to(directory).parts)
            except ValueError:
                continue
            if relative_depth > _MAX_SCAN_DEPTH or not path.is_file():
                continue
            if suffixes and path.suffix.lower() not in suffixes and not any(path.name.lower().endswith(item) for item in suffixes):
                continue
            if names and path.name not in names:
                continue
            result.append(path)
            if len(result) >= _MAX_ITEMS_PER_RESOURCE:
                break
    except OSError:
        return result
    return sorted(result, key=lambda item: str(item).lower())


def _item(path: Path, root: Path, kind: str, *, scope: str, active: bool, status: str | None = None, **extra: Any) -> dict[str, Any]:
    try:
        relative = str(path.relative_to(root))
    except ValueError:
        relative = path.name
    result: dict[str, Any] = {
        "name": path.stem,
        "kind": kind,
        "path": str(path),
        "relativePath": relative,
        "scope": scope,
        "source": "workspace" if scope in {"project", "local"} else "agent",
        "active": bool(active),
        "status": status or ("active" if active else "detected_only"),
    }
    result.update(extra)
    return result


def _frontmatter_metadata(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(_MAX_FRONTMATTER_BYTES)
    except OSError:
        return {}
    text = raw.decode("utf-8", errors="replace")
    if not text.startswith("---"):
        return {}
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = next((index for index, line in enumerate(lines[1:], start=1) if line.strip() in {"---", "..."}), None)
    if end is None:
        return {}
    try:
        value = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _claude_imports_agents(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            text = handle.read(_MAX_FRONTMATTER_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return False
    return "@AGENTS.md" in text or "@./AGENTS.md" in text


def _summarize_settings(path: Path) -> dict[str, Any]:
    value = _read_json_object(path)
    if value is None:
        return {
            "valid": False,
            "hookEvents": [],
            "hasPermissionConfig": False,
            "permissionRules": {effect: [] for effect in _PERMISSION_EFFECTS},
            "permissionRuleCounts": {},
            "permissionDefaultMode": "",
            "disableBypassPermissionsMode": False,
            "enabledMcpServers": [],
        }
    permissions = value.get("permissions") if isinstance(value.get("permissions"), Mapping) else {}
    rules = {effect: _string_list(permissions.get(effect)) for effect in _PERMISSION_EFFECTS}
    counts = {effect: len(rules[effect]) for effect in _PERMISSION_EFFECTS}
    default_mode = str(permissions.get("defaultMode") or "").strip()
    hooks = value.get("hooks") if isinstance(value.get("hooks"), Mapping) else {}
    enabled_servers = value.get("enabledMcpjsonServers")
    return {
        "valid": True,
        "hookEvents": sorted(str(key) for key in hooks if str(key).strip()),
        "hasPermissionConfig": bool(
            permissions
            and (
                any(counts.values())
                or default_mode
                or "disableBypassPermissionsMode" in permissions
            )
        ),
        "permissionRules": rules,
        "permissionRuleCounts": counts,
        "permissionDefaultMode": default_mode,
        "disableBypassPermissionsMode": (
            str(permissions.get("disableBypassPermissionsMode") or "").strip() == "disable"
        ),
        "enabledMcpServers": _string_list(enabled_servers),
        "enableAllProjectMcpServers": bool(value.get("enableAllProjectMcpServers", False)),
        "hasEnableAllProjectMcpServers": "enableAllProjectMcpServers" in value,
    }


def _permission_item_fields(settings_item: Mapping[str, Any]) -> dict[str, Any]:
    rules = settings_item.get("permissionRules")
    return {
        "ruleCounts": dict(settings_item.get("permissionRuleCounts") or {}),
        "rules": {
            effect: list(rules.get(effect) or []) if isinstance(rules, Mapping) else []
            for effect in _PERMISSION_EFFECTS
        },
        "defaultMode": str(settings_item.get("permissionDefaultMode") or ""),
        "disableBypassPermissionsMode": bool(settings_item.get("disableBypassPermissionsMode")),
    }


def _summarize_mcp(path: Path) -> dict[str, Any]:
    value = _read_json_object(path)
    if value is None:
        return {"valid": False, "serverNames": []}
    servers = value.get("mcpServers") if isinstance(value.get("mcpServers"), Mapping) else {}
    return {
        "valid": True,
        "serverNames": [str(key) for key in servers if str(key).strip()],
    }


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as handle:
            raw = handle.read(_MAX_JSON_BYTES + 1)
        if len(raw) > _MAX_JSON_BYTES:
            return None
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _activate_additional_memory(additional_scans: Sequence[Mapping[str, Any]], resources: dict[str, Any]) -> None:
    additional_paths = {
        str(item.get("path"))
        for scan in additional_scans
        for item in scan.get("memory", [])
        if isinstance(item, Mapping)
    }
    memory = resources.get("memory")
    if not isinstance(memory, Mapping):
        return
    for item in memory.get("items", []):
        if isinstance(item, dict) and str(item.get("path")) in additional_paths:
            item["active"] = True
            item["status"] = "active"
    memory["activeCount"] = sum(1 for item in memory.get("items", []) if isinstance(item, Mapping) and item.get("active"))


def _find_conflicts(
    *,
    commands: Mapping[str, Any],
    mcp: Mapping[str, Any],
    permission: Mapping[str, Any],
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    command_items = commands.get("items", []) if isinstance(commands.get("items"), list) else []
    names: dict[str, list[Mapping[str, Any]]] = {}
    for item in command_items:
        if isinstance(item, Mapping):
            names.setdefault(str(item.get("name") or "").lower(), []).append(item)
    for name, items in names.items():
        if len(items) > 1 and name:
            kinds = {str(item.get("kind") or "") for item in items}
            conflicts.append({
                "type": "command_name_collision",
                "name": name,
                "paths": [str(item.get("path") or "") for item in items],
                "resolution": (
                    "skill_precedes_command_when_active"
                    if {"skill", "command"}.issubset(kinds)
                    else "claude_code_native_scope_precedence"
                ),
            })
    for name in mcp.get("nameCollisions", []):
        conflicts.append({
            "type": "mcp_name_collision",
            "name": name,
            "resolution": "claude_code_native_mcp_resolution",
        })
    if permission.get("agentDisallowedTools") and permission.get("workspaceRulesActive"):
        conflicts.append({
            "type": "permission_cap",
            "resolution": "agent_disallowed_tools_remain_denied",
            "tools": list(permission["agentDisallowedTools"]),
        })
    return conflicts


def _fingerprint(resources: Mapping[str, Any], roots: Sequence[WorkspaceRoot], agent_scans: Sequence[Mapping[str, Any]]) -> str:
    values: list[str] = [str(root.path) for root in roots]
    for group in (resources, {"agent": agent_scans}):
        for item in _iter_items(group):
            path_text = str(item.get("path") or "")
            if not path_text:
                continue
            try:
                stat = os.stat(path_text)
                values.append(f"{path_text}:{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                values.append(f"{path_text}:missing")
    digest = hashlib.sha256("\n".join(sorted(set(values))).encode("utf-8")).hexdigest()
    return digest[:24]


def _iter_items(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"items", "memory", "settings", "mcp", "hooks", "permissions", "skills", "commands", "outputStyles", "agents", "workflows"} and isinstance(item, list):
                for child in item:
                    if isinstance(child, Mapping):
                        yield child
            yield from _iter_items(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_items(item)


def _normalize_observed_command(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {
            key: _safe_observed_value(value[key])
            for key in ("name", "description", "argumentHint", "source", "type")
            if key in value
        }
    return {"name": str(value)}


def _normalize_observed_mcp_status(value: Mapping[str, Any]) -> dict[str, Any]:
    servers = value.get("mcpServers")
    if not isinstance(servers, list):
        return {"mcpServers": []}
    result: list[dict[str, Any]] = []
    for item in servers[:_MAX_ITEMS_PER_RESOURCE]:
        if not isinstance(item, Mapping):
            continue
        server: dict[str, Any] = {
            key: _safe_observed_value(item[key])
            for key in ("name", "status", "scope", "error", "serverInfo")
            if key in item
        }
        tools = item.get("tools")
        if isinstance(tools, list):
            server["tools"] = [
                {
                    key: _safe_observed_value(tool[key])
                    for key in ("name", "description", "annotations")
                    if key in tool
                }
                for tool in tools[:_MAX_ITEMS_PER_RESOURCE]
                if isinstance(tool, Mapping)
            ]
        result.append(server)
    return {"mcpServers": result}


def _safe_observed_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_observed_value(item) for key, item in value.items() if str(key).lower() not in {"token", "password", "secret", "api_key", "apikey", "authorization", "env"}}
    if isinstance(value, list):
        return [_safe_observed_value(item) for item in value[:_MAX_ITEMS_PER_RESOURCE]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _deep_copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))
