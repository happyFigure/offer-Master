from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import shutil


@dataclass(slots=True)
class ServerSettings:
    host: str
    port: int


@dataclass(slots=True)
class ClaudeSettings:
    workdir: Path
    config_dir: Path
    default_model: str
    permission_mode: str
    cli_path: str | None
    setting_sources: list[str] | None
    skills_filter: list[str] | str | None
    system_prompt_preset: str | None
    system_prompt_append: str
    system_prompt_file: Path | None
    include_hook_events: bool
    enable_file_checkpointing: bool
    attachment_text_char_limit: int
    tools: Any | None = None
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    strict_mcp_config: bool = False
    continue_conversation: bool = False
    max_turns: int | None = None
    max_budget_usd: float | None = None
    fallback_model: str | None = None
    betas: list[str] | None = None
    permission_prompt_tool_name: str | None = None
    settings: str | None = None
    add_dirs: list[Path] | None = None
    env: dict[str, str] | None = None
    extra_args: dict[str, str | None] | None = None
    max_buffer_size: int | None = None
    user: str | None = None
    include_partial_messages: bool = True
    fork_session: bool = False
    agents: dict[str, Any] | None = None
    sandbox: dict[str, Any] | None = None
    plugins: list[dict[str, Any]] | None = None
    max_thinking_tokens: int | None = None
    thinking: dict[str, Any] | None = None
    effort: str | None = None
    output_format: dict[str, Any] | None = None
    session_store_flush: str | None = None
    load_timeout_ms: int | None = None
    task_budget: dict[str, Any] | None = None


@dataclass(slots=True)
class ProviderSettings:
    base_url: str
    anthropic_version: str
    api_key: str
    request_timeout_sec: float
    proxy_context_ttl_sec: float = 259200.0


@dataclass(slots=True)
class AuthSettings:
    enabled: bool
    uac_auth_url: str
    allow_users_path: Path
    shared_tdl_api_key_path: Path


@dataclass(slots=True)
class SessionSettings:
    mapping_path: Path
    checkpoints_path: Path
    goals_path: Path | None = None


@dataclass(slots=True)
class SkillSettings:
    source_dirs: list[Path]
    mount_dir: Path


@dataclass(slots=True)
class SkillUsageAuditSettings:
    enabled: bool
    base_url: str
    endpoint: str
    timeout_sec: float


@dataclass(slots=True)
class WorkflowSettings:
    source_dirs: list[Path]
    target_dir: Path


@dataclass(slots=True)
class McpSettings:
    config_dir: Path
    auto_load: bool
    extra_config_dirs: list[Path] | None = None


@dataclass(slots=True)
class FeatureSettings:
    auto_interrupt_on_disconnect: bool
    approval_frontend_enabled: bool
    question_frontend_enabled: bool
    hook_frontend_enabled: bool
    checkpoint_rewind_frontend_enabled: bool
    task_panel_frontend_enabled: bool
    subagent_events_frontend_enabled: bool


@dataclass(slots=True)
class AppSettings:
    root: Path
    server: ServerSettings
    claude: ClaudeSettings
    provider: ProviderSettings
    auth: AuthSettings
    sessions: SessionSettings
    skills: SkillSettings
    skill_usage_audit: SkillUsageAuditSettings
    workflows: WorkflowSettings
    mcp: McpSettings
    features: FeatureSettings


def _resolve_path(root: Path, raw: str, *, default: str) -> Path:
    value = str(raw or default).strip() or default
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _load_claude_settings_json(config_dir: Path) -> dict[str, Any]:
    settings_path = config_dir / "settings.json"
    if not settings_path.exists():
        template_path = config_dir / "settings.template.json"
        if not template_path.exists():
            return {}
        try:
            shutil.copyfile(template_path, settings_path)
        except Exception:
            return {}
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _optional_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    items = [str(item or "").strip() for item in value]
    cleaned = [item for item in items if item]
    return cleaned if cleaned else []


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value or "").strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _optional_mapping_list(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    items = [dict(item) for item in value if isinstance(item, dict)]
    return items if items else []


def _optional_env(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    env: dict[str, str] = {}
    for key, item in value.items():
        name = str(key or "").strip()
        if not name or item is None:
            continue
        env[name] = str(item)
    return env if env else {}


def _optional_extra_args(value: Any) -> dict[str, str | None] | None:
    if not isinstance(value, dict):
        return None
    args: dict[str, str | None] = {}
    for key, item in value.items():
        name = str(key or "").strip().lstrip("-")
        if not name:
            continue
        args[name] = None if item is None else str(item)
    return args if args else {}


def _optional_path_list(root: Path, value: Any) -> list[Path] | None:
    if not isinstance(value, list):
        return None
    paths: list[Path] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            paths.append(_resolve_path(root, text, default=text))
    return paths if paths else []


def _optional_tools(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return [str(item or "").strip() for item in value if str(item or "").strip()]
    text = str(value or "").strip()
    if text == "claude_code":
        return {"type": "preset", "preset": "claude_code"}
    return None


def _resolve_cli_path(raw: str | None) -> str | None:
    candidate = str(raw or "").strip()
    if candidate:
        return candidate
    candidates = _claude_cli_candidates()
    versioned: list[tuple[tuple[int, ...], int, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(candidates):
        text = str(item or "").strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if not path.exists() or not path.is_file():
            continue
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        version = _claude_cli_version(resolved)
        if version:
            versioned.append((version, -index, resolved))
    if versioned:
        return max(versioned)[2]
    # Leave the path unset when every discovered candidate is broken. The
    # Agent SDK can then use its bundled CLI instead of an unusable shell shim.
    return None


def _claude_cli_candidates() -> list[str | None]:
    home = Path.home()
    paths: list[str | None] = [
        shutil.which("claude"),
        str(home / ".npm-global" / "bin" / "claude"),
        str(home / ".local" / "bin" / "claude"),
        str(home / "bin" / "claude"),
    ]
    opt_root = home / ".local" / "opt"
    if opt_root.exists():
        for item in sorted(opt_root.glob("node-*/bin/claude")):
            paths.append(str(item))
    paths.extend(["/usr/local/bin/claude", "/usr/bin/claude"])
    return paths


def _claude_cli_version(path: str) -> tuple[int, ...]:
    try:
        result = subprocess.run(
            [path, "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if result.returncode != 0:
        return ()
    match = re.search(r"(\d+(?:\.\d+)+)", result.stdout or "")
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def load_settings(root: Path) -> AppSettings:
    config_path = root / "config" / "service.json"
    data: dict[str, Any] = {}
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))

    server_data = data.get("server") if isinstance(data.get("server"), dict) else {}
    claude_data = data.get("claude") if isinstance(data.get("claude"), dict) else {}
    auth_data = data.get("auth") if isinstance(data.get("auth"), dict) else {}
    session_data = data.get("sessions") if isinstance(data.get("sessions"), dict) else {}
    skills_data = data.get("skills") if isinstance(data.get("skills"), dict) else {}
    workflows_data = data.get("workflows") if isinstance(data.get("workflows"), dict) else {}
    mcp_data = data.get("mcp") if isinstance(data.get("mcp"), dict) else {}
    feature_data = data.get("features") if isinstance(data.get("features"), dict) else {}

    server = ServerSettings(
        host=os.getenv("CLAUDE_SDK_AGENT_HOST", str(server_data.get("host") or "0.0.0.0")),
        port=int(os.getenv("CLAUDE_SDK_AGENT_PORT", str(server_data.get("port") or 18008))),
    )
    config_dir = _resolve_path(root, str(os.getenv("CLAUDE_SDK_AGENT_CONFIG_DIR") or claude_data.get("config_dir") or "../claude-code/.claude"), default="../claude-code/.claude")
    claude_settings_json = _load_claude_settings_json(config_dir)
    claude_env = claude_settings_json.get("env") if isinstance(claude_settings_json.get("env"), dict) else {}
    claude = ClaudeSettings(
        workdir=_resolve_path(root, str(os.getenv("CLAUDE_SDK_AGENT_WORKDIR") or claude_data.get("workdir") or "../.."), default="../.."),
        config_dir=config_dir,
        default_model=str(os.getenv("CLAUDE_SDK_AGENT_MODEL") or claude_data.get("default_model") or "MiniMax-M2.7").strip(),
        permission_mode=str(os.getenv("CLAUDE_SDK_AGENT_PERMISSION_MODE") or claude_data.get("permission_mode") or "bypassPermissions").strip(),
        cli_path=_resolve_cli_path(str(os.getenv("CLAUDE_SDK_AGENT_CLI_PATH") or claude_data.get("cli_path") or "")),
        setting_sources=_optional_str_list(claude_data.get("setting_sources")),
        skills_filter=(
            "all"
            if str(os.getenv("CLAUDE_SDK_AGENT_SKILLS_FILTER") or claude_data.get("skills_filter") or "").strip().lower() == "all"
            else _optional_str_list(claude_data.get("skills_filter"))
        ),
        system_prompt_preset=str(os.getenv("CLAUDE_SDK_AGENT_SYSTEM_PROMPT_PRESET") or claude_data.get("system_prompt_preset") or "").strip() or None,
        system_prompt_append=str(os.getenv("CLAUDE_SDK_AGENT_SYSTEM_PROMPT_APPEND") or claude_data.get("system_prompt_append") or "").strip(),
        system_prompt_file=(
            _resolve_path(
                root,
                str(os.getenv("CLAUDE_SDK_AGENT_SYSTEM_PROMPT_FILE") or claude_data.get("system_prompt_file") or ""),
                default=str(os.getenv("CLAUDE_SDK_AGENT_SYSTEM_PROMPT_FILE") or claude_data.get("system_prompt_file") or ""),
            )
            if str(os.getenv("CLAUDE_SDK_AGENT_SYSTEM_PROMPT_FILE") or claude_data.get("system_prompt_file") or "").strip()
            else None
        ),
        include_hook_events=_env_bool(
            "CLAUDE_SDK_AGENT_INCLUDE_HOOK_EVENTS",
            bool(claude_data.get("include_hook_events", False)),
        ),
        enable_file_checkpointing=_env_bool(
            "CLAUDE_SDK_AGENT_ENABLE_FILE_CHECKPOINTING",
            bool(claude_data.get("enable_file_checkpointing", False)),
        ),
        attachment_text_char_limit=max(
            4096,
            _env_int(
                "CLAUDE_SDK_AGENT_ATTACHMENT_TEXT_CHAR_LIMIT",
                int(claude_data.get("attachment_text_char_limit") or 256000),
            ),
        ),
        tools=_optional_tools(claude_data.get("tools")),
        allowed_tools=_optional_str_list(claude_data.get("allowed_tools")),
        disallowed_tools=_optional_str_list(claude_data.get("disallowed_tools")),
        strict_mcp_config=_env_bool(
            "CLAUDE_SDK_AGENT_STRICT_MCP_CONFIG",
            bool(claude_data.get("strict_mcp_config", False)),
        ),
        continue_conversation=_env_bool(
            "CLAUDE_SDK_AGENT_CONTINUE_CONVERSATION",
            bool(claude_data.get("continue_conversation", False)),
        ),
        max_turns=_optional_int(os.getenv("CLAUDE_SDK_AGENT_MAX_TURNS") or claude_data.get("max_turns")),
        max_budget_usd=_optional_float(os.getenv("CLAUDE_SDK_AGENT_MAX_BUDGET_USD") or claude_data.get("max_budget_usd")),
        fallback_model=_optional_str(os.getenv("CLAUDE_SDK_AGENT_FALLBACK_MODEL") or claude_data.get("fallback_model")),
        betas=_optional_str_list(claude_data.get("betas")),
        permission_prompt_tool_name=_optional_str(
            os.getenv("CLAUDE_SDK_AGENT_PERMISSION_PROMPT_TOOL_NAME") or claude_data.get("permission_prompt_tool_name")
        ),
        settings=(
            str(
                _resolve_path(
                    root,
                    str(os.getenv("CLAUDE_SDK_AGENT_SETTINGS") or claude_data.get("settings") or ""),
                    default=str(os.getenv("CLAUDE_SDK_AGENT_SETTINGS") or claude_data.get("settings") or ""),
                )
            )
            if str(os.getenv("CLAUDE_SDK_AGENT_SETTINGS") or claude_data.get("settings") or "").strip()
            else None
        ),
        add_dirs=_optional_path_list(root, claude_data.get("add_dirs")),
        env=_optional_env(claude_data.get("env")),
        extra_args=_optional_extra_args(claude_data.get("extra_args")),
        max_buffer_size=_optional_int(os.getenv("CLAUDE_SDK_AGENT_MAX_BUFFER_SIZE") or claude_data.get("max_buffer_size")),
        user=_optional_str(os.getenv("CLAUDE_SDK_AGENT_USER") or claude_data.get("user")),
        include_partial_messages=_env_bool(
            "CLAUDE_SDK_AGENT_INCLUDE_PARTIAL_MESSAGES",
            bool(claude_data.get("include_partial_messages", True)),
        ),
        fork_session=_env_bool(
            "CLAUDE_SDK_AGENT_FORK_SESSION",
            bool(claude_data.get("fork_session", False)),
        ),
        agents=_optional_mapping(claude_data.get("agents")),
        sandbox=_optional_mapping(claude_data.get("sandbox")),
        plugins=_optional_mapping_list(claude_data.get("plugins")),
        max_thinking_tokens=_optional_int(
            os.getenv("CLAUDE_SDK_AGENT_MAX_THINKING_TOKENS") or claude_data.get("max_thinking_tokens")
        ),
        thinking=_optional_mapping(claude_data.get("thinking")),
        effort=_optional_str(os.getenv("CLAUDE_SDK_AGENT_EFFORT") or claude_data.get("effort")),
        output_format=_optional_mapping(claude_data.get("output_format")),
        session_store_flush=_optional_str(claude_data.get("session_store_flush")),
        load_timeout_ms=_optional_int(os.getenv("CLAUDE_SDK_AGENT_LOAD_TIMEOUT_MS") or claude_data.get("load_timeout_ms")),
        task_budget=_optional_mapping(claude_data.get("task_budget")),
    )
    provider_data = data.get("provider") if isinstance(data.get("provider"), dict) else {}
    provider = ProviderSettings(
        base_url=str(os.getenv("CLAUDE_SDK_AGENT_PROVIDER_BASE_URL") or provider_data.get("base_url") or claude_env.get("ANTHROPIC_BASE_URL") or "").strip(),
        anthropic_version=str(os.getenv("CLAUDE_SDK_AGENT_ANTHROPIC_VERSION") or provider_data.get("anthropic_version") or "2023-06-01").strip(),
        api_key=str(os.getenv("CLAUDE_SDK_AGENT_PROVIDER_API_KEY") or provider_data.get("api_key") or claude_env.get("ANTHROPIC_AUTH_TOKEN") or "").strip(),
        request_timeout_sec=float(os.getenv("CLAUDE_SDK_AGENT_PROVIDER_TIMEOUT_SEC") or provider_data.get("request_timeout_sec") or 600.0),
        proxy_context_ttl_sec=float(
            os.getenv("CLAUDE_SDK_AGENT_PROXY_CONTEXT_TTL_SEC")
            or provider_data.get("proxy_context_ttl_sec")
            or 259200.0
        ),
    )
    auth = AuthSettings(
        enabled=_env_bool("CLAUDE_SDK_AGENT_AUTH_ENABLED", bool(auth_data.get("enabled", True))),
        uac_auth_url=str(os.getenv("CLAUDE_SDK_AGENT_UAC_AUTH_URL") or auth_data.get("uac_auth_url") or "").strip(),
        allow_users_path=_resolve_path(root, str(os.getenv("CLAUDE_SDK_AGENT_ALLOW_USERS_PATH") or auth_data.get("allow_users_path") or "config/allow_users.json"), default="config/allow_users.json"),
        shared_tdl_api_key_path=_resolve_path(
            root,
            str(
                os.getenv("CLAUDE_SDK_AGENT_SHARED_TDL_API_KEY_PATH")
                or auth_data.get("shared_tdl_api_key_path")
                or "../my-agents/config/allow_users.json"
            ),
            default="../my-agents/config/allow_users.json",
        ),
    )
    sessions = SessionSettings(
        mapping_path=_resolve_path(root, str(os.getenv("CLAUDE_SDK_AGENT_MAPPING_PATH") or session_data.get("mapping_path") or "data/sessions/session-map.json"), default="data/sessions/session-map.json"),
        checkpoints_path=_resolve_path(
            root,
            str(os.getenv("CLAUDE_SDK_AGENT_CHECKPOINTS_PATH") or session_data.get("checkpoints_path") or "data/sessions/checkpoints.json"),
            default="data/sessions/checkpoints.json",
        ),
        goals_path=_resolve_path(
            root,
            str(os.getenv("CLAUDE_SDK_AGENT_GOALS_PATH") or session_data.get("goals_path") or "data/sessions/goals.json"),
            default="data/sessions/goals.json",
        ),
    )
    raw_skill_dirs = skills_data.get("source_dirs")
    source_dirs: list[Path] = []
    if isinstance(raw_skill_dirs, list):
        for item in raw_skill_dirs:
            text = str(item or "").strip()
            if text:
                source_dirs.append(_resolve_path(root, text, default=text))
    skills = SkillSettings(
        source_dirs=source_dirs,
        mount_dir=_resolve_path(
            root,
            str(os.getenv("CLAUDE_SDK_AGENT_SKILL_MOUNT_DIR") or skills_data.get("mount_dir") or "data/skill-mount"),
            default="data/skill-mount",
        ),
    )
    skill_usage_data = data.get("skill_usage_audit") if isinstance(data.get("skill_usage_audit"), dict) else {}
    skill_usage_audit = SkillUsageAuditSettings(
        enabled=bool(skill_usage_data.get("enabled", True)),
        base_url=str(skill_usage_data.get("base_url") or "http://127.0.0.1:18000").strip().rstrip("/"),
        endpoint=str(skill_usage_data.get("endpoint") or "/v1/skill-center/usage").strip()
        or "/v1/skill-center/usage",
        timeout_sec=float(skill_usage_data.get("timeout_sec") or 3.0),
    )
    raw_workflow_dirs = workflows_data.get("source_dirs")
    workflow_source_dirs: list[Path] = []
    if isinstance(raw_workflow_dirs, list):
        for item in raw_workflow_dirs:
            text = str(item or "").strip()
            if text:
                workflow_source_dirs.append(_resolve_path(root, text, default=text))
    default_workflow_target = str(claude.config_dir / "workflows")
    workflows = WorkflowSettings(
        source_dirs=workflow_source_dirs,
        target_dir=_resolve_path(
            root,
            str(
                os.getenv("CLAUDE_SDK_AGENT_WORKFLOW_TARGET_DIR")
                or workflows_data.get("target_dir")
                or default_workflow_target
            ),
            default=default_workflow_target,
        ),
    )
    mcp = McpSettings(
        config_dir=_resolve_path(
            root,
            str(os.getenv("CLAUDE_SDK_AGENT_MCP_CONFIG_DIR") or mcp_data.get("config_dir") or "../my-agents/mcps"),
            default="../my-agents/mcps",
        ),
        extra_config_dirs=[
            _resolve_path(root, str(item), default=str(item))
            for item in (_optional_str_list(mcp_data.get("extra_config_dirs")) or [])
        ],
        auto_load=_env_bool(
            "CLAUDE_SDK_AGENT_MCP_AUTO_LOAD",
            bool(mcp_data.get("auto_load", True)),
        ),
    )
    features = FeatureSettings(
        auto_interrupt_on_disconnect=_env_bool(
            "CLAUDE_SDK_AGENT_AUTO_INTERRUPT_ON_DISCONNECT",
            bool(feature_data.get("auto_interrupt_on_disconnect", True)),
        ),
        approval_frontend_enabled=_env_bool(
            "CLAUDE_SDK_AGENT_APPROVAL_FRONTEND_ENABLED",
            bool(feature_data.get("approval_frontend_enabled", False)),
        ),
        question_frontend_enabled=_env_bool(
            "CLAUDE_SDK_AGENT_QUESTION_FRONTEND_ENABLED",
            bool(feature_data.get("question_frontend_enabled", False)),
        ),
        hook_frontend_enabled=_env_bool(
            "CLAUDE_SDK_AGENT_HOOK_FRONTEND_ENABLED",
            bool(feature_data.get("hook_frontend_enabled", False)),
        ),
        checkpoint_rewind_frontend_enabled=_env_bool(
            "CLAUDE_SDK_AGENT_CHECKPOINT_REWIND_FRONTEND_ENABLED",
            bool(feature_data.get("checkpoint_rewind_frontend_enabled", False)),
        ),
        task_panel_frontend_enabled=_env_bool(
            "CLAUDE_SDK_AGENT_TASK_PANEL_FRONTEND_ENABLED",
            bool(feature_data.get("task_panel_frontend_enabled", False)),
        ),
        subagent_events_frontend_enabled=_env_bool(
            "CLAUDE_SDK_AGENT_SUBAGENT_EVENTS_FRONTEND_ENABLED",
            bool(feature_data.get("subagent_events_frontend_enabled", False)),
        ),
    )
    return AppSettings(
        root=root,
        server=server,
        claude=claude,
        provider=provider,
        auth=auth,
        sessions=sessions,
        skills=skills,
        skill_usage_audit=skill_usage_audit,
        workflows=workflows,
        mcp=mcp,
        features=features,
    )
