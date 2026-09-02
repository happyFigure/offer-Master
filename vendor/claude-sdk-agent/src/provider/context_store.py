from __future__ import annotations

import asyncio
import secrets
import time

from .models import ProxyContext


class ProxyContextStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._items: dict[str, ProxyContext] = {}

    async def create(
        self,
        *,
        upstream_base_url: str,
        anthropic_version: str,
        x_user_id: str,
        api_token: str,
        model: str = "",
        request_headers: dict[str, str],
        ttl_sec: float = 3600.0,
    ) -> ProxyContext:
        token = secrets.token_urlsafe(24)
        ttl = max(1.0, float(ttl_sec or 3600.0))
        ctx = ProxyContext(
            proxy_token=token,
            upstream_base_url=upstream_base_url.rstrip("/"),
            anthropic_version=anthropic_version,
            x_user_id=x_user_id,
            api_token=api_token,
            model=str(model or "").strip(),
            request_headers=dict(request_headers),
            ttl_sec=ttl,
            expires_at=time.time() + ttl,
        )
        async with self._lock:
            self._purge_locked()
            self._items[token] = ctx
        return ctx

    async def get(self, token: str) -> ProxyContext | None:
        async with self._lock:
            self._purge_locked()
            ctx = self._items.get(token)
            if ctx is not None:
                ctx.expires_at = time.time() + max(1.0, float(ctx.ttl_sec or 3600.0))
            return ctx

    def _purge_locked(self) -> None:
        now = time.time()
        expired = [token for token, ctx in self._items.items() if ctx.expires_at <= now]
        for token in expired:
            self._items.pop(token, None)
