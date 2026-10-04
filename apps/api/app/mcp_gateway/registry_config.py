from __future__ import annotations

"""声明式 MCP 注册配置。

配置文件只描述如何启动 MCP 以及允许暴露哪些工具；账号、密码等敏感值
通过 ``${ENV_NAME}`` 从运行时环境读取，永远不写入注册表或日志。
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


@dataclass(frozen=True)
class MCPServerConfig:
    id: str
    tool_prefix: str
    transport: str = "stdio"
    command: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    timeout_seconds: float = 30.0
    enabled: bool = True
    enabled_env: str | None = None
    allow_tools: tuple[str, ...] = ()
    deny_tools: tuple[str, ...] = ()
    policy: str = "default"

    def __post_init__(self) -> None:
        server_id = str(self.id or "").strip().lower()
        prefix = str(self.tool_prefix or "").strip().strip(".").lower()
        transport = str(self.transport or "stdio").strip().lower()
        if not server_id:
            raise ValueError("MCP server id is required")
        if not prefix:
            raise ValueError(f"MCP server tool_prefix is required: {server_id}")
        if transport != "stdio":
            raise ValueError(f"Unsupported MCP transport: {transport}")
        if not str(self.command or "").strip():
            raise ValueError(f"MCP server command is required: {server_id}")
        object.__setattr__(self, "id", server_id)
        object.__setattr__(self, "tool_prefix", prefix)
        object.__setattr__(self, "transport", transport)
        object.__setattr__(self, "args", tuple(str(value) for value in self.args))
        object.__setattr__(self, "env", {str(key): str(value) for key, value in dict(self.env or {}).items()})
        object.__setattr__(self, "allow_tools", _unique_strings(self.allow_tools))
        object.__setattr__(self, "deny_tools", _unique_strings(self.deny_tools))
        object.__setattr__(self, "timeout_seconds", max(1.0, float(self.timeout_seconds or 30.0)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, environ: Mapping[str, str] | None = None) -> MCPServerConfig:
        values = dict(payload)
        env = dict(environ or os.environ)
        raw_enabled = values.get("enabled", True)
        enabled = bool(raw_enabled)
        enabled_env = values.get("enabled_env")
        if enabled_env:
            enabled = enabled and _env_flag(env.get(str(enabled_env)))
        # Disabled optional connectors must not make application startup fail
        # because their credentials are intentionally absent.
        resolved_env = (
            {str(key): resolve_env_references(value, environ=env) for key, value in dict(values.get("env") or {}).items()}
            if enabled
            else {}
        )
        return cls(
            id=resolve_env_references(values.get("id", ""), environ=env),
            tool_prefix=resolve_env_references(values.get("tool_prefix", ""), environ=env),
            transport=resolve_env_references(values.get("transport", "stdio"), environ=env),
            command=resolve_env_references(values.get("command", ""), environ=env),
            args=tuple(resolve_env_references(value, environ=env) for value in values.get("args", ()) or ()),
            env=resolved_env,
            cwd=resolve_env_references(values.get("cwd"), environ=env) if values.get("cwd") else None,
            timeout_seconds=float(values.get("timeout_seconds", 30.0) or 30.0),
            enabled=enabled,
            enabled_env=str(enabled_env).strip() if enabled_env else None,
            allow_tools=tuple(resolve_env_references(value, environ=env) for value in values.get("allow_tools", ()) or ()),
            deny_tools=tuple(resolve_env_references(value, environ=env) for value in values.get("deny_tools", ()) or ()),
            policy=resolve_env_references(values.get("policy", "default"), environ=env),
        )


def load_mcp_server_configs(path: Path, *, environ: Mapping[str, str] | None = None) -> tuple[MCPServerConfig, ...]:
    if not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("servers", []) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("MCP registry config must contain a 'servers' list")
    configs = tuple(
        MCPServerConfig.from_dict(entry, environ=environ)
        for entry in entries
        if isinstance(entry, dict)
    )
    ids = [config.id for config in configs]
    if len(ids) != len(set(ids)):
        raise ValueError("MCP registry server ids must be unique")
    return tuple(config for config in configs if config.enabled)


def resolve_env_references(value: Any, *, environ: Mapping[str, str] | None = None) -> Any:
    if not isinstance(value, str):
        return value
    env = environ or os.environ

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        resolved = env.get(name)
        if resolved is None or resolved == "":
            if default is not None:
                return default
            raise ValueError(f"Required MCP environment variable is missing: {name}")
        return resolved

    return _ENV_REFERENCE.sub(replace, value)


def _env_flag(value: str | None) -> bool:
    return str(value or "").strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _unique_strings(values: Any) -> tuple[str, ...]:
    result: list[str] = []
    for value in values or ():
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


__all__ = ["MCPServerConfig", "load_mcp_server_configs", "resolve_env_references"]
