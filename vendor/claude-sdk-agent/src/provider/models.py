from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(slots=True)
class ProxyContext:
    proxy_token: str
    upstream_base_url: str
    anthropic_version: str
    x_user_id: str
    api_token: str
    model: str = ""
    request_headers: Mapping[str, str] = field(default_factory=dict)
    ttl_sec: float = 3600.0
    expires_at: float = 0.0
