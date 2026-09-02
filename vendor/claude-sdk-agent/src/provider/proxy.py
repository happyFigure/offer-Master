from __future__ import annotations

import contextlib
import fnmatch
import ipaddress
import json
import logging
import os
from typing import AsyncIterator, Mapping
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from starlette.background import BackgroundTask
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..config import ProviderSettings
from .context_store import ProxyContextStore
from .models import ProxyContext

logger = logging.getLogger(__name__)

_CLAUDE_INTERNAL_MODEL_PREFIXES = ("claude-",)
_DEFAULT_NO_PROXY = "127.0.0.1,::1,localhost,10.0.0.0/8,.zte.com.cn,zte.com.cn"


def install_provider_routes(app: FastAPI, provider: ProviderSettings, store: ProxyContextStore) -> None:
    @app.api_route("/internal/anthropic", methods=["GET", "HEAD"])
    @app.api_route("/internal/anthropic/", methods=["GET", "HEAD"])
    async def anthropic_proxy_root() -> Response:
        return Response(status_code=200)

    @app.api_route("/internal/anthropic/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def anthropic_proxy(path: str, request: Request) -> Response:
        proxy_token = _extract_proxy_token(request.headers)
        if not proxy_token:
            return JSONResponse({"error": {"message": "missing proxy token"}}, status_code=401)
        ctx = await store.get(proxy_token)
        if ctx is None:
            return JSONResponse({"error": {"message": "invalid or expired proxy token"}}, status_code=401)
        upstream_url = f"{ctx.upstream_base_url}/{path.lstrip('/')}"
        headers = _build_upstream_headers(request.headers, ctx)
        body = await request.body()
        body = _rewrite_internal_model_body(path, body, ctx)
        wants_stream = _request_wants_stream(request, body)
        logger.info(
            "[proxy] request method=%s path=%s upstream=%s stream=%s x_user_id=%s",
            request.method,
            path,
            upstream_url,
            wants_stream,
            ctx.x_user_id or "-",
        )
        if wants_stream:
            client_cm = _make_proxy_client(provider.request_timeout_sec, upstream_url=upstream_url)
            client = await client_cm.__aenter__()
            response = await client.send(
                client.build_request(request.method, upstream_url, content=body, headers=headers),
                stream=True,
            )
            content_type = response.headers.get("content-type", "")
            logger.info(
                "[proxy] upstream stream response status=%s content_type=%s path=%s",
                response.status_code,
                content_type or "-",
                path,
            )
            if "text/event-stream" not in content_type.lower():
                try:
                    content = await response.aread()
                    if response.status_code >= 400:
                        logger.warning(
                            "[proxy] upstream stream error status=%s path=%s body=%s",
                            response.status_code,
                            path,
                            _safe_log_body(content),
                        )
                    return Response(content=content, status_code=response.status_code, media_type=content_type or None)
                finally:
                    await response.aclose()
                    await client_cm.__aexit__(None, None, None)
            return StreamingResponse(
                _stream_upstream_response(response, client_cm),
                status_code=response.status_code,
                media_type=content_type or "text/event-stream",
                background=BackgroundTask(_close_response_and_client, response, client_cm),
            )
        async with _make_proxy_client(provider.request_timeout_sec, upstream_url=upstream_url) as client:
            response = await client.send(
                client.build_request(request.method, upstream_url, content=body, headers=headers),
                stream=True,
            )
            try:
                content = await _read_raw_response_content(response)
                content_type = response.headers.get("content-type", "")
                logger.info(
                    "[proxy] upstream response status=%s content_type=%s path=%s",
                    response.status_code,
                    content_type or "-",
                    path,
                )
                if response.status_code >= 400:
                    logger.warning(
                        "[proxy] upstream error status=%s path=%s body=%s",
                        response.status_code,
                        path,
                        _safe_log_body(content),
                    )
                return Response(
                    content=content,
                    status_code=response.status_code,
                    media_type=content_type or None,
                )
            finally:
                await response.aclose()


def _extract_proxy_token(headers: Mapping[str, str]) -> str:
    for key in ("x-api-key", "api-key", "authorization"):
        value = headers.get(key)
        if not value:
            continue
        text = str(value).strip()
        if not text:
            continue
        if key == "authorization" and text.lower().startswith("bearer "):
            return text[7:].strip()
        return text
    return ""


def _request_wants_stream(request: Request, body: bytes) -> bool:
    accept = str(request.headers.get("accept") or "").lower()
    if "text/event-stream" in accept:
        return True
    content_type = str(request.headers.get("content-type") or "").lower()
    if "application/json" not in content_type:
        return False
    if not body:
        return False
    try:
        payload = json.loads(body)
    except Exception:
        return False
    return isinstance(payload, dict) and payload.get("stream") is True


def _rewrite_internal_model_body(path: str, body: bytes, ctx: ProxyContext) -> bytes:
    main_model = str(ctx.model or "").strip()
    if not main_model or not body or not _path_accepts_model_rewrite(path):
        return body
    try:
        payload = json.loads(body)
    except Exception:
        return body
    if not isinstance(payload, dict):
        return body
    requested_model = str(payload.get("model") or "").strip()
    if not _should_rewrite_internal_model(requested_model, main_model):
        return body
    payload["model"] = main_model
    logger.info(
        "[proxy] rewrite internal model path=%s model=%s -> %s",
        path,
        requested_model or "<empty>",
        main_model,
    )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _path_accepts_model_rewrite(path: str) -> bool:
    normalized = "/" + str(path or "").strip().lstrip("/")
    return normalized in {"/v1/messages", "/v1/messages/count_tokens"}


def _should_rewrite_internal_model(requested_model: str, main_model: str) -> bool:
    requested = str(requested_model or "").strip()
    target = str(main_model or "").strip()
    if not target or requested == target:
        return False
    if not requested:
        return True
    return requested.lower().startswith(_CLAUDE_INTERNAL_MODEL_PREFIXES)


def _build_upstream_headers(request_headers: Mapping[str, str], ctx: ProxyContext) -> dict[str, str]:
    out = {
        "content-type": request_headers.get("content-type", "application/json"),
        "anthropic-version": request_headers.get("anthropic-version", ctx.anthropic_version),
        "x-user-id": ctx.x_user_id,
        "X-User-Id": ctx.x_user_id,
    }
    accept = request_headers.get("accept")
    if accept:
        out["accept"] = accept
    out["accept-encoding"] = "identity"
    beta = request_headers.get("anthropic-beta")
    if beta:
        out["anthropic-beta"] = beta
    forwarded = ctx.request_headers
    for key in ("uac-user-id", "x-uac-user-id", "uac-user-token", "x-uac-user-token"):
        value = forwarded.get(key)
        if value:
            out[key] = value
    if ctx.api_token:
        out["authorization"] = f"Bearer {ctx.api_token}"
        out["x-api-key"] = ctx.api_token
        out["api-key"] = ctx.api_token
    return out


def _safe_log_body(content: bytes, *, limit: int = 2000) -> str:
    text = content.decode("utf-8", errors="replace").strip()
    if len(text) > limit:
        return f"{text[:limit]}...<truncated:{len(text) - limit} chars>"
    return text


@contextlib.asynccontextmanager
async def _make_proxy_client(timeout_sec: float, *, upstream_url: str = ""):
    async with httpx.AsyncClient(
        timeout=timeout_sec,
        follow_redirects=True,
        proxy=_proxy_url_for_upstream(upstream_url),
        trust_env=False,
    ) as client:
        yield client


def _proxy_url_for_upstream(upstream_url: str) -> str | None:
    parsed = urlsplit(str(upstream_url or ""))
    scheme = parsed.scheme.lower()
    host = str(parsed.hostname or "").strip()
    no_proxy = ",".join(
        item for item in (_DEFAULT_NO_PROXY, _env_value("NO_PROXY", "no_proxy")) if item
    )
    if not scheme or not host or _host_matches_no_proxy(host, no_proxy):
        return None
    candidates = (
        ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")
        if scheme == "https"
        else ("HTTP_PROXY", "http_proxy")
    )
    for name in candidates:
        value = str(os.environ.get(name) or "").strip()
        if value.startswith(("http://", "https://")):
            return value
    return None


def _env_value(*names: str) -> str:
    for name in names:
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def _host_matches_no_proxy(host: str, no_proxy: str) -> bool:
    normalized_host = str(host or "").strip().lower().strip("[]")
    if not normalized_host:
        return False
    try:
        host_ip = ipaddress.ip_address(normalized_host)
    except ValueError:
        host_ip = None
    for raw_item in str(no_proxy or "").split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item == "*":
            return True
        if host_ip is not None and _ip_matches_no_proxy(host_ip, item):
            return True
        if fnmatch.fnmatch(normalized_host, item):
            return True
        suffix = item[1:] if item.startswith(".") else item
        if normalized_host == suffix or normalized_host.endswith(f".{suffix}"):
            return True
    return False


def _ip_matches_no_proxy(host_ip: ipaddress._BaseAddress, item: str) -> bool:
    try:
        if "/" in item:
            return host_ip in ipaddress.ip_network(item, strict=False)
        return host_ip == ipaddress.ip_address(item)
    except ValueError:
        return False


async def _stream_upstream_response(response: httpx.Response, client_cm) -> AsyncIterator[bytes]:
    chunk_count = 0
    async for chunk in response.aiter_raw():
        if chunk:
            chunk_count += 1
            yield chunk
    logger.info("[proxy] upstream stream finished status=%s chunks=%s", response.status_code, chunk_count)


async def _read_raw_response_content(response: httpx.Response) -> bytes:
    chunks: list[bytes] = []
    async for chunk in response.aiter_raw():
        if chunk:
            chunks.append(chunk)
    return b"".join(chunks)


async def _close_response_and_client(response: httpx.Response, client_cm) -> None:
    await response.aclose()
    await client_cm.__aexit__(None, None, None)
