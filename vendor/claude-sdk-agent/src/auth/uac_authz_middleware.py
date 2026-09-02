from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, Mapping

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..api.openai_compat import completion_payload, new_chat_id, single_text_sse
from ..config import AppSettings
from .models import CurrentUser

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_AUTH_BYPASS_PATHS = frozenset({"/healthz", "/v1/runtime/status"})
_AUTH_BYPASS_PREFIXES = ("/internal/anthropic",)


def _looks_like_openai_chat_path(path: str) -> bool:
    return (path or "").lower().endswith("/v1/chat/completions")


def _is_loopback_request(request: Request) -> bool:
    client = request.client
    return (client.host if client else "").strip() in _LOOPBACK_HOSTS


def _should_bypass_auth(request: Request) -> bool:
    if request.method.upper() == "OPTIONS":
        return True
    path = request.url.path
    if any(path.startswith(prefix) for prefix in _AUTH_BYPASS_PREFIXES):
        return True
    return request.method.upper() == "GET" and request.url.path in _AUTH_BYPASS_PATHS and _is_loopback_request(request)


def _get_header_value(headers: Mapping[str, str], *candidates: str) -> str:
    normalized = {candidate.strip().lower().replace("_", "-") for candidate in candidates if candidate.strip()}
    for key, value in headers.items():
        if str(key or "").strip().lower().replace("_", "-") in normalized:
            text = str(value or "").strip()
            if text:
                return text
    return ""


def _get_tdl_api_key(headers: Mapping[str, str]) -> str:
    return _get_header_value(headers, "api-key", "x-api-key", "api_key", "x_api_key")


def _looks_like_tdl_api_key(api_key: str) -> bool:
    return str(api_key or "").strip().startswith("tdl_")


def _extract_user_id_from_auth_response(response_data: Any) -> str | None:
    if not isinstance(response_data, dict):
        return None
    if response_data.get("result", False) is not True:
        return None
    data = response_data.get("data", {})
    if not isinstance(data, dict):
        return None
    returned_user_id = str(data.get("user_id") or "").strip()
    return returned_user_id or None


async def _single_text_stream(message: str, model_name: str) -> AsyncIterator[bytes]:
    chat_id = new_chat_id()
    async for chunk in single_text_sse(chat_id=chat_id, model=model_name, content=message):
        yield chunk


async def _resolve_stream_flag_for_request(request: Request) -> bool:
    if not _looks_like_openai_chat_path(request.url.path):
        return False
    try:
        raw = await request.body()
        if raw:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload.get("stream") is not False
    except Exception:
        pass
    return True


async def _authz_failure_response(request: Request, *, message: str, status_code: int, model_name: str) -> JSONResponse | StreamingResponse:
    if _looks_like_openai_chat_path(request.url.path):
        if await _resolve_stream_flag_for_request(request):
            return StreamingResponse(_single_text_stream(message, model_name), media_type="text/event-stream")
        return JSONResponse(completion_payload(chat_id=new_chat_id(), model=model_name, content=message), status_code=200)
    return JSONResponse(status_code=status_code, content={"result": False, "failReason": message})


async def _verify_tdl_api_key_for_request(request: Request, api_key: str, auth_url: str) -> str | None:
    cache_ttl_sec = 300.0
    verify_timeout_sec = 15.0
    auth_cache = getattr(request.app.state, "_tdl_api_key_auth_cache", None)
    if auth_cache is None:
        auth_cache = {}
        setattr(request.app.state, "_tdl_api_key_auth_cache", auth_cache)
    now_ts = time.time()
    entry = auth_cache.get(api_key)
    if isinstance(entry, dict):
        cached_user_id = str(entry.get("user_id") or "").strip()
        expires_at = float(entry.get("expires_at", 0))
        if cached_user_id and expires_at > now_ts:
            return cached_user_id
        auth_cache.pop(api_key, None)
    try:
        async with httpx.AsyncClient(timeout=verify_timeout_sec, follow_redirects=True, proxy=None, trust_env=False) as client:
            response = await client.post(auth_url, json={"api-key": api_key})
        if response.status_code != 200:
            return None
        user_id = _extract_user_id_from_auth_response(response.json())
        if user_id:
            auth_cache[api_key] = {"user_id": user_id, "expires_at": now_ts + cache_ttl_sec}
        return user_id
    except Exception as exc:
        logger.warning("[authz] TDL API Key verify failed: %s", exc)
        return None


def _sync_tdl_api_key_env(api_key: str, user_id: str) -> None:
    os.environ["TDL_API_KEY"] = api_key
    os.environ["USER"] = user_id
    os.environ["coclaw_empno"] = user_id
    os.environ["RDCLOUD_EMP_NO"] = user_id


def _load_shared_tdl_api_key(request: Request, path: Path) -> str:
    cache = getattr(request.app.state, "_shared_tdl_api_key_cache", None)
    if cache is None:
        cache = {"mtime": None, "value": ""}
        setattr(request.app.state, "_shared_tdl_api_key_cache", cache)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        cache["mtime"] = None
        cache["value"] = ""
        return ""
    if cache.get("mtime") == mtime:
        return str(cache.get("value") or "").strip()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        cache["mtime"] = mtime
        cache["value"] = ""
        return ""
    value = ""
    if isinstance(data, dict):
        value = str(data.get("x-api-key") or data.get("api-key") or "").strip()
    cache["mtime"] = mtime
    cache["value"] = value
    return value


async def _validate_allow_users(request: Request, *, allow_users_path: Path, allow_users_path_str: str, user_id: str, model_name: str) -> JSONResponse | StreamingResponse | None:
    allow_users_cache = getattr(request.app.state, "_allow_users_cache", None)
    if allow_users_cache is None:
        allow_users_cache = {"mtime": None, "allow_all": True, "allowed_set": set()}
        setattr(request.app.state, "_allow_users_cache", allow_users_cache)
    try:
        allow_users_mtime = allow_users_path.stat().st_mtime
    except FileNotFoundError:
        allow_users_path.parent.mkdir(parents=True, exist_ok=True)
        allow_users_path.write_text(json.dumps({"allow_users": []}, ensure_ascii=False, indent=2), encoding="utf-8")
        allow_users_mtime = allow_users_path.stat().st_mtime
    initialized_from_empty_list = False
    if allow_users_cache.get("mtime") != allow_users_mtime:
        allow_all = True
        allowed_set: set[str] = set()
        text = allow_users_path.read_text(encoding="utf-8").strip() if allow_users_path.exists() else ""
        if text:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                allowed_set = {str(item).strip() for item in parsed if str(item).strip()}
            elif isinstance(parsed, dict):
                maybe = parsed.get("allow_users")
                if isinstance(maybe, list):
                    if len(maybe) == 0:
                        allowed_set = {str(user_id).strip()}
                        allow_all = False
                        initialized_from_empty_list = True
                        payload = dict(parsed)
                        payload["allow_users"] = [str(user_id).strip()]
                        allow_users_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    else:
                        allowed_set = {str(item).strip() for item in maybe if str(item).strip()}
            if allowed_set:
                allow_all = False
        allow_users_cache["mtime"] = allow_users_mtime
        allow_users_cache["allow_all"] = allow_all
        allow_users_cache["allowed_set"] = allowed_set
    allow_all_current_request = True if initialized_from_empty_list else bool(allow_users_cache.get("allow_all", True))
    if not allow_all_current_request:
        allowed_set = allow_users_cache.get("allowed_set", set())
        if str(user_id).strip() not in allowed_set:
            return await _authz_failure_response(
                request,
                status_code=403,
                model_name=model_name,
                message=(
                    "权限不足：当前 user_id 不在允许列表。"
                    f"请在该IP的个人电脑修改允许列表：{allow_users_path_str}，添加允许访问的 user_id。"
                ),
            )
    return None


def install_uac_authz_inject_env_middleware(app: FastAPI, settings: AppSettings) -> None:
    allow_users_path = settings.auth.allow_users_path
    allow_users_path_str = str(allow_users_path)
    model_name = settings.claude.default_model
    shared_tdl_api_key_path = settings.auth.shared_tdl_api_key_path

    @app.middleware("http")
    async def _uac_authz_and_inject_env_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        request.state.shared_tdl_api_key = _load_shared_tdl_api_key(request, shared_tdl_api_key_path)
        if not settings.auth.enabled or _should_bypass_auth(request):
            return await call_next(request)

        tdl_api_key = _get_tdl_api_key(request.headers)
        if tdl_api_key and _looks_like_tdl_api_key(tdl_api_key):
            user_id = await _verify_tdl_api_key_for_request(request, tdl_api_key, settings.auth.uac_auth_url)
            if not user_id:
                return await _authz_failure_response(
                    request,
                    status_code=401,
                    model_name=model_name,
                    message=(
                        "鉴权失败：TDL API Key 校验不通过。"
                        f"请在该IP的个人电脑修改允许列表：{allow_users_path_str}，添加允许访问的 user_id。"
                    ),
                )
            allow_failure = await _validate_allow_users(
                request,
                allow_users_path=allow_users_path,
                allow_users_path_str=allow_users_path_str,
                user_id=user_id,
                model_name=model_name,
            )
            if allow_failure is not None:
                return allow_failure
            _sync_tdl_api_key_env(tdl_api_key, user_id)
            request.state.current_user = CurrentUser(name=user_id, emp_id=user_id)
            return await call_next(request)

        uac_user_id = str(request.headers.get("uac-user-id") or request.headers.get("x-uac-user-id") or "").strip()
        uac_user_token = str(request.headers.get("uac-user-token") or request.headers.get("x-uac-user-token") or "").strip()
        if not uac_user_id or not uac_user_token:
            return await _authz_failure_response(
                request,
                status_code=401,
                model_name=model_name,
                message=(
                    "鉴权失败：缺少请求头 uac-user-id/uac-user-token。"
                    f"请在该IP的个人电脑修改允许列表：{allow_users_path_str}，添加允许访问的 user_id。"
                ),
            )

        verify_timeout_sec = 15.0
        cache_ttl_sec = 300.0
        uac_auth_cache = getattr(request.app.state, "_uac_auth_cache", None)
        if uac_auth_cache is None:
            uac_auth_cache = {}
            setattr(request.app.state, "_uac_auth_cache", uac_auth_cache)
        now_ts = time.time()
        entry = uac_auth_cache.get(uac_user_token)
        uac_ok_user_id: str | None = None
        if isinstance(entry, dict):
            cached_user_id = str(entry.get("user_id") or "").strip()
            expires_at = float(entry.get("expires_at", 0))
            if expires_at > now_ts and cached_user_id == uac_user_id:
                uac_ok_user_id = uac_user_id
            elif expires_at <= now_ts:
                uac_auth_cache.pop(uac_user_token, None)

        async def _verify_uac_token() -> str | None:
            payload_candidates = [
                {"api-key/token": uac_user_token, "user-id": uac_user_id},
                {"api-key/token": uac_user_token, "user_id": uac_user_id},
                {"api-key": uac_user_token, "user-id": uac_user_id},
                {"api-key": uac_user_token, "user_id": uac_user_id},
                {"x-api-key/token": uac_user_token, "user-id": uac_user_id},
                {"x-api-key/token": uac_user_token, "user_id": uac_user_id},
                {"x-api-key": uac_user_token, "user-id": uac_user_id},
                {"x-api-key": uac_user_token, "user_id": uac_user_id},
            ]
            async with httpx.AsyncClient(timeout=verify_timeout_sec, follow_redirects=True, proxy=None, trust_env=False) as client:
                for body in payload_candidates:
                    try:
                        response = await client.post(settings.auth.uac_auth_url, json=body)
                    except Exception:
                        continue
                    if response.status_code != 200:
                        continue
                    returned_user_id = _extract_user_id_from_auth_response(response.json())
                    if returned_user_id and returned_user_id == uac_user_id:
                        return returned_user_id
            return None

        if uac_ok_user_id is None:
            uac_ok_user_id = await _verify_uac_token()
            if not uac_ok_user_id:
                return await _authz_failure_response(
                    request,
                    status_code=401,
                    model_name=model_name,
                    message=(
                        "鉴权失败：UAC 校验不通过。"
                        f"请在该IP的个人电脑修改允许列表：{allow_users_path_str}，添加允许访问的 user_id。"
                    ),
                )
            uac_auth_cache[uac_user_token] = {"user_id": uac_ok_user_id, "expires_at": now_ts + cache_ttl_sec}

        allow_failure = await _validate_allow_users(
            request,
            allow_users_path=allow_users_path,
            allow_users_path_str=allow_users_path_str,
            user_id=uac_ok_user_id,
            model_name=model_name,
        )
        if allow_failure is not None:
            return allow_failure
        os.environ["USER"] = uac_ok_user_id
        os.environ["coclaw_empno"] = uac_ok_user_id
        os.environ["RDCLOUD_EMP_NO"] = uac_ok_user_id
        request.state.current_user = CurrentUser(name=uac_ok_user_id, emp_id=uac_ok_user_id)
        return await call_next(request)
