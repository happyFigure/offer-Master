from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .config import McpSettings


def build_mcp_servers(
    settings: McpSettings,
    *,
    request_headers: Mapping[str, str],
    fallback_tdl_api_key: str,
) -> dict[str, dict[str, Any]]:
    if not settings.auto_load:
        return {}
    servers: dict[str, dict[str, Any]] = {}
    config_dirs = [settings.config_dir, *getattr(settings, "extra_config_dirs", [])]
    for config_dir in config_dirs:
        if not config_dir.exists() or not config_dir.is_dir():
            continue
        for path in sorted(config_dir.glob("*.json")):
            server = _load_mcp_server(path, request_headers=request_headers, fallback_tdl_api_key=fallback_tdl_api_key)
            if server:
                name, config = server
                servers[name] = config
    return servers


def _load_mcp_server(
    path: Path,
    *,
    request_headers: Mapping[str, str],
    fallback_tdl_api_key: str,
) -> tuple[str, dict[str, Any]] | None:
        if path.name.endswith(".example.json"):
            return None
        payload = _load_json(path)
        if not isinstance(payload, dict):
            return None
        name = str(payload.get("name") or path.stem).strip()
        transport = str(payload.get("transport") or "").strip().lower()
        if not transport and payload.get("command"):
            transport = "stdio"
        if not name:
            return None
        if transport in {"http", "sse"}:
            url = str(payload.get("url") or "").strip()
            if not url:
                return None
            headers = _resolve_headers(payload.get("headers"), request_headers, fallback_tdl_api_key=fallback_tdl_api_key)
            server: dict[str, Any] = {
                "type": "sse" if transport == "sse" else "http",
                "url": url,
            }
            if headers:
                server["headers"] = headers
        elif transport == "stdio":
            command = str(payload.get("command") or "").strip()
            if not command:
                return None
            server = {"command": command}
            args = _optional_str_list(payload.get("args"))
            env = _resolve_env(payload.get("env"), request_headers, fallback_tdl_api_key=fallback_tdl_api_key)
            if args:
                server["args"] = args
            if env:
                server["env"] = env
        else:
            return None
        return name, server


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_headers(raw_headers: Any, request_headers: Mapping[str, str], *, fallback_tdl_api_key: str) -> dict[str, str]:
    if not isinstance(raw_headers, dict):
        raw_headers = {}
    resolved: dict[str, str] = {}
    request_tdl_key = _first_non_empty(request_headers.get("x-api-key"), request_headers.get("api-key"))
    effective_tdl_key = request_tdl_key or str(fallback_tdl_api_key or "").strip()
    request_uac_id = _first_non_empty(request_headers.get("uac-user-id"), request_headers.get("x-uac-user-id"))
    request_uac_token = _first_non_empty(request_headers.get("uac-user-token"), request_headers.get("x-uac-user-token"))
    for key, value in raw_headers.items():
        header_name = str(key or "").strip()
        if not header_name:
            continue
        header_value = str(value or "").strip()
        lower_name = header_name.lower()
        if lower_name in {"x-api-key", "api-key"} and not header_value:
            header_value = effective_tdl_key
        elif lower_name in {"uac-user-id", "x-uac-user-id"} and not header_value:
            header_value = request_uac_id
        elif lower_name in {"uac-user-token", "x-uac-user-token"} and not header_value:
            header_value = request_uac_token
        if header_value:
            resolved[header_name] = header_value
    if effective_tdl_key and "x-api-key" not in {key.lower() for key in resolved} and "api-key" not in {key.lower() for key in resolved}:
        resolved["x-api-key"] = effective_tdl_key
    return resolved


def _resolve_env(raw_env: Any, request_headers: Mapping[str, str], *, fallback_tdl_api_key: str) -> dict[str, str]:
    if not isinstance(raw_env, dict):
        return {}
    request_tdl_key = _first_non_empty(request_headers.get("x-api-key"), request_headers.get("api-key"))
    effective_tdl_key = request_tdl_key or str(fallback_tdl_api_key or "").strip()
    resolved: dict[str, str] = {}
    for key, value in raw_env.items():
        env_name = str(key or "").strip()
        if not env_name:
            continue
        env_value = str(value or "").strip()
        if not env_value and env_name in {"MINIMAX_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "TDL_API_KEY"}:
            env_value = effective_tdl_key
        if env_value:
            resolved[env_name] = env_value
    return resolved


def _optional_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item or "").strip() for item in value if str(item or "").strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""
