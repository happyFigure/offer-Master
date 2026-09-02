from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.server import create_app
from src.provider.context_store import ProxyContextStore
from src.provider.proxy import _proxy_url_for_upstream, _request_wants_stream


def _write_service_json(root: Path) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "data" / "sessions").mkdir(parents=True, exist_ok=True)
    (root / "config" / "allow_users.json").write_text(json.dumps({"allow_users": []}), encoding="utf-8")
    payload = {
        "server": {"host": "127.0.0.1", "port": 18008},
        "claude": {
            "workdir": ".",
            "config_dir": ".",
            "default_model": "MiniMax-RAN3",
            "permission_mode": "acceptEdits",
        },
        "provider": {
            "base_url": "http://upstream.example",
            "anthropic_version": "2023-06-01",
            "request_timeout_sec": 30.0,
        },
        "auth": {
            "enabled": False,
            "uac_auth_url": "http://127.0.0.1:9998/auth",
            "allow_users_path": "config/allow_users.json",
        },
        "sessions": {"mapping_path": "data/sessions/session-map.json"},
    }
    (root / "config" / "service.json").write_text(json.dumps(payload), encoding="utf-8")


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, headers: dict[str, str] | None = None, content: bytes = b"{}") -> None:
        self.status_code = status_code
        self.headers = headers or {"content-type": "application/json"}
        self.content = content

    async def aiter_raw(self):
        yield self.content

    async def aclose(self) -> None:
        return None


class _FakeClient:
    def __init__(self, recorder: dict[str, Any]) -> None:
        self.recorder = recorder

    def build_request(self, method: str, url: str, content: bytes, headers: dict[str, str]):
        return {
            "method": method,
            "url": url,
            "content": content,
            "headers": headers,
        }

    async def request(self, method: str, url: str, content: bytes, headers: dict[str, str]):
        self.recorder["method"] = method
        self.recorder["url"] = url
        self.recorder["content"] = content
        self.recorder["headers"] = headers
        return _FakeResponse(content=b'{"ok":true}')

    async def send(self, request, *, stream: bool = False):
        self.recorder["method"] = request["method"]
        self.recorder["url"] = request["url"]
        self.recorder["content"] = request["content"]
        self.recorder["headers"] = request["headers"]
        self.recorder["stream"] = stream
        return _FakeResponse(content=b'{"ok":true}')


class _FakeClientCm:
    def __init__(self, recorder: dict[str, Any]) -> None:
        self.client = _FakeClient(recorder)

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, exc_type, exc, tb):
        return False


class ProxyTests(unittest.TestCase):
    def test_builtin_intranet_upstreams_bypass_proxy_without_no_proxy_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HTTPS_PROXY": "http://proxy.example:80",
                "HTTP_PROXY": "http://proxy-http.example:80",
            },
            clear=True,
        ):
            self.assertIsNone(_proxy_url_for_upstream("http://10.2.67.41:18081/v1/messages"))
            self.assertIsNone(_proxy_url_for_upstream("https://wxai-icf.zx.zte.com.cn/v1/messages"))
            self.assertIsNone(_proxy_url_for_upstream("http://localhost:18008/internal/anthropic"))
            self.assertIsNone(_proxy_url_for_upstream("http://127.0.0.1:18008/internal/anthropic"))
            self.assertIsNone(_proxy_url_for_upstream("http://[::1]:18008/internal/anthropic"))
            self.assertEqual(
                _proxy_url_for_upstream("https://api.krill-ai.com/v1/messages"),
                "http://proxy.example:80",
            )

    def test_external_upstream_uses_https_proxy_without_all_proxy(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HTTPS_PROXY": "http://proxy.example:80",
                "HTTP_PROXY": "http://proxy-http.example:80",
                "ALL_PROXY": "socks://proxy.example:80",
                "NO_PROXY": "127.0.0.1,localhost,10.0.0.0/8,.zte.com.cn",
            },
            clear=True,
        ):
            self.assertEqual(
                _proxy_url_for_upstream("https://api.krill-ai.com/v1/messages"),
                "http://proxy.example:80",
            )
            self.assertIsNone(_proxy_url_for_upstream("http://10.2.67.41:18081/v1/messages"))
            self.assertIsNone(_proxy_url_for_upstream("https://wxai-icf.zx.zte.com.cn/v1/messages"))

    def test_internal_proxy_root_handles_head_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root)
            app = create_app(root=root)
            client = TestClient(app)

            response = client.head("/internal/anthropic")
            self.assertEqual(response.status_code, 200)

    def test_request_wants_stream_when_json_body_enables_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root)
            app = create_app(root=root)
            client = TestClient(app)
            request = client.build_request(
                "POST",
                "/internal/anthropic/v1/messages",
                headers={"content-type": "application/json"},
                json={"model": "MiniMax-RAN3", "stream": True},
            )
            body = json.dumps({"model": "MiniMax-RAN3", "stream": True}).encode("utf-8")
            scope = {
                "type": "http",
                "method": "POST",
                "path": "/internal/anthropic/v1/messages",
                "headers": request.headers.raw,
                "client": ("10.137.58.137", 12345),
                "scheme": "http",
                "server": ("10.137.58.137", 18008),
                "query_string": b"",
                "root_path": "",
                "http_version": "1.1",
            }
            from starlette.requests import Request as StarletteRequest

            self.assertTrue(_request_wants_stream(StarletteRequest(scope), body))

    def test_internal_proxy_injects_user_headers_and_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root)
            app = create_app(root=root)
            ctx = asyncio.run(
                app.state.proxy_contexts.create(
                    upstream_base_url="http://upstream.example",
                    anthropic_version="2023-06-01",
                    x_user_id="10154402",
                    api_token="real-token",
                    model="MiniMax-RAN3",
                    request_headers={
                        "uac-user-id": "10154402",
                        "uac-user-token": "uac-token",
                        "x-api-key": "real-token",
                    },
                )
            )
            recorder: dict[str, Any] = {}
            client = TestClient(app)
            with patch("src.provider.proxy._make_proxy_client", return_value=_FakeClientCm(recorder)):
                response = client.post(
                    "/internal/anthropic/v1/messages",
                    headers={"x-api-key": ctx.proxy_token, "content-type": "application/json"},
                    json={"model": "MiniMax-RAN3"},
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(recorder["url"], "http://upstream.example/v1/messages")
            self.assertEqual(recorder["headers"]["X-User-Id"], "10154402")
            self.assertEqual(recorder["headers"]["uac-user-token"], "uac-token")
            self.assertEqual(recorder["headers"]["authorization"], "Bearer real-token")
            self.assertEqual(recorder["headers"]["accept-encoding"], "identity")

    def test_proxy_context_get_extends_expiry_for_long_running_sessions(self) -> None:
        async def scenario() -> tuple[float, float]:
            store = ProxyContextStore()
            ctx = await store.create(
                upstream_base_url="http://upstream.example",
                anthropic_version="2023-06-01",
                x_user_id="10154402",
                api_token="real-token",
                model="MiniMax-RAN3",
                request_headers={},
                ttl_sec=120.0,
            )
            original_expiry = ctx.expires_at
            ctx.expires_at = time.time() + 1.0
            refreshed = await store.get(ctx.proxy_token)
            self.assertIsNotNone(refreshed)
            return original_expiry, refreshed.expires_at if refreshed is not None else 0.0

        original_expiry, refreshed_expiry = asyncio.run(scenario())
        self.assertGreater(refreshed_expiry, original_expiry)

    def test_internal_proxy_rewrites_claude_child_model_to_main_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root)
            app = create_app(root=root)
            ctx = asyncio.run(
                app.state.proxy_contexts.create(
                    upstream_base_url="http://upstream.example",
                    anthropic_version="2023-06-01",
                    x_user_id="10154402",
                    api_token="real-token",
                    model="MiniMax-RAN3",
                    request_headers={},
                )
            )
            recorder: dict[str, Any] = {}
            client = TestClient(app)
            with patch("src.provider.proxy._make_proxy_client", return_value=_FakeClientCm(recorder)):
                response = client.post(
                    "/internal/anthropic/v1/messages",
                    headers={"x-api-key": ctx.proxy_token, "content-type": "application/json"},
                    json={"model": "claude-opus-4-8", "stream": False, "messages": []},
                )

            self.assertEqual(response.status_code, 200)
            forwarded = json.loads(recorder["content"].decode("utf-8"))
            self.assertEqual(forwarded["model"], "MiniMax-RAN3")

    def test_internal_proxy_keeps_custom_non_claude_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_service_json(root)
            app = create_app(root=root)
            ctx = asyncio.run(
                app.state.proxy_contexts.create(
                    upstream_base_url="http://upstream.example",
                    anthropic_version="2023-06-01",
                    x_user_id="10154402",
                    api_token="real-token",
                    model="MiniMax-RAN3",
                    request_headers={},
                )
            )
            recorder: dict[str, Any] = {}
            client = TestClient(app)
            with patch("src.provider.proxy._make_proxy_client", return_value=_FakeClientCm(recorder)):
                response = client.post(
                    "/internal/anthropic/v1/messages",
                    headers={"x-api-key": ctx.proxy_token, "content-type": "application/json"},
                    json={"model": "MiniMax-RAN3-Custom", "messages": []},
                )

            self.assertEqual(response.status_code, 200)
            forwarded = json.loads(recorder["content"].decode("utf-8"))
            self.assertEqual(forwarded["model"], "MiniMax-RAN3-Custom")
