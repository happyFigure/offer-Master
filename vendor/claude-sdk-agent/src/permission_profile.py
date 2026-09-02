from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import ClaudeSettings


PROFILE_READONLY = "readonly"
PROFILE_SAFE = "safe"
PROFILE_EDIT = "edit"
PROFILE_BYPASS = "bypass"
PROFILE_AUTO = "auto"
PROFILE_DONT_ASK = "dontAsk"
PROFILE_FULL_BYPASS = "fullBypass"
PERMISSION_PROFILES = {
    PROFILE_READONLY,
    PROFILE_SAFE,
    PROFILE_EDIT,
    PROFILE_BYPASS,
    PROFILE_AUTO,
    PROFILE_DONT_ASK,
    PROFILE_FULL_BYPASS,
}
NATIVE_PERMISSION_MODES = {
    "default",
    "acceptEdits",
    "plan",
    "auto",
    "dontAsk",
    "bypassPermissions",
}
PROFILE_PERMISSION_MODES = {
    PROFILE_READONLY: "plan",
    PROFILE_SAFE: "default",
    PROFILE_EDIT: "acceptEdits",
    PROFILE_BYPASS: "bypassPermissions",
    PROFILE_AUTO: "auto",
    PROFILE_DONT_ASK: "dontAsk",
    PROFILE_FULL_BYPASS: "bypassPermissions",
}
MODE_PROFILES = {
    "default": PROFILE_SAFE,
    "acceptEdits": PROFILE_EDIT,
    "plan": PROFILE_READONLY,
    "auto": PROFILE_AUTO,
    "dontAsk": PROFILE_DONT_ASK,
    "bypassPermissions": PROFILE_BYPASS,
}
PERMISSION_PROFILE_ITEMS = (
    {
        "id": PROFILE_READONLY,
        "title": "规划模式",
        "description": "使用 Claude Code 原生 plan 模式；规划阶段只读，批准 ExitPlanMode 后会退出规划并按后续权限执行。",
        "permissionMode": "plan",
    },
    {
        "id": PROFILE_SAFE,
        "title": "安全确认",
        "description": "使用默认确认流，需要时由前端审批或追问。",
        "permissionMode": "default",
    },
    {
        "id": PROFILE_EDIT,
        "title": "允许常规编辑",
        "description": "自动允许常规文件编辑，危险操作仍走确认。",
        "permissionMode": "acceptEdits",
    },
    {
        "id": PROFILE_AUTO,
        "title": "自动判断",
        "description": "由 Claude Code 安全分类器判断工具调用。",
        "permissionMode": "auto",
    },
    {
        "id": PROFILE_DONT_ASK,
        "title": "仅运行已批准操作",
        "description": "使用 dontAsk，需要确认的操作直接拒绝。",
        "permissionMode": "dontAsk",
    },
    {
        "id": PROFILE_BYPASS,
        "title": "Claude Code 原生绕过",
        "description": "保留 Agent 工具限制，使用 Claude Code 原生 bypassPermissions。",
        "permissionMode": "bypassPermissions",
    },
    {
        "id": PROFILE_FULL_BYPASS,
        "title": "平台最高权限",
        "description": "清除 Agent 工具限制，并请求 Claude Code 原生危险绕过模式。",
        "permissionMode": "bypassPermissions",
        "fullBypass": True,
    },
)


@dataclass(frozen=True, slots=True)
class PermissionOptions:
    profile: str
    permission_mode: str
    allowed_tools: list[str] | None
    disallowed_tools: list[str] | None
    full_bypass: bool = False
    runtime_key: str = ""
    revision: int = 0
    updated_at: int | None = None
    updated_by: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "permissionMode": self.permission_mode,
            "allowedTools": self.allowed_tools,
            "disallowedTools": self.disallowed_tools,
            "fullBypass": self.full_bypass,
            "runtimeKey": self.runtime_key,
            "revision": self.revision,
            "updatedAt": self.updated_at,
            "updatedBy": self.updated_by,
        }


def default_permission_profile(settings: ClaudeSettings) -> str:
    mode = str(settings.permission_mode or "").strip()
    return MODE_PROFILES.get(mode, PROFILE_SAFE)


def effective_permission_options(
    settings: ClaudeSettings,
    profile: str,
    *,
    runtime_key: str = "",
    revision: int = 0,
    updated_at: int | None = None,
    updated_by: str = "",
) -> PermissionOptions:
    normalized = normalize_permission_profile(profile)
    return PermissionOptions(
        profile=normalized,
        permission_mode=PROFILE_PERMISSION_MODES[normalized],
        allowed_tools=None if normalized == PROFILE_FULL_BYPASS else _unique_strings(settings.allowed_tools),
        disallowed_tools=None if normalized == PROFILE_FULL_BYPASS else _unique_strings(settings.disallowed_tools),
        full_bypass=normalized == PROFILE_FULL_BYPASS,
        runtime_key=str(runtime_key or "").strip(),
        revision=_non_negative_int(revision),
        updated_at=updated_at,
        updated_by=str(updated_by or "").strip(),
    )


def permission_options_from_runtime_config(
    settings: ClaudeSettings,
    config: Mapping[str, Any] | None,
    *,
    fallback_runtime_key: str = "",
) -> PermissionOptions:
    section = config if isinstance(config, Mapping) else {}
    raw_profile = str(section.get("profile") or "").strip()
    try:
        profile = normalize_permission_profile(raw_profile) if raw_profile else default_permission_profile(settings)
    except ValueError:
        profile = default_permission_profile(settings)
    return effective_permission_options(
        settings,
        profile,
        runtime_key=str(
            section.get("runtime_key")
            or section.get("runtimeKey")
            or fallback_runtime_key
            or ""
        ).strip(),
        revision=_non_negative_int(section.get("revision")),
        updated_at=_optional_int(section.get("updated_at") or section.get("updatedAt")),
        updated_by=str(section.get("updated_by") or section.get("updatedBy") or "").strip(),
    )


def permission_snapshot(
    settings: ClaudeSettings,
    config: Mapping[str, Any] | None = None,
    *,
    fallback_runtime_key: str = "",
    source: str = "request_runtime_config",
) -> dict[str, Any]:
    options = permission_options_from_runtime_config(
        settings,
        config,
        fallback_runtime_key=fallback_runtime_key,
    )
    return {
        "supported": True,
        "profiles": [dict(item) for item in PERMISSION_PROFILE_ITEMS],
        "current": options.to_payload(),
        "source": source,
        "runtimeKey": options.runtime_key,
        "revision": options.revision,
        "scope": "request",
    }


def normalize_permission_profile(value: str) -> str:
    text = str(value or "").strip()
    if text in NATIVE_PERMISSION_MODES:
        return MODE_PROFILES[text]
    if text not in PERMISSION_PROFILES:
        raise ValueError(f"Unsupported permission profile: {text or '-'}")
    return text


def _unique_strings(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
