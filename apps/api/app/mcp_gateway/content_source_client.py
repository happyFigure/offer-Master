from __future__ import annotations

from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from app.mcp_gateway.client import MCPGatewayClientProtocol, MCPToolCallResult


class ContentSourceClientProtocol(Protocol):
    def read_weixin_article(self, *, url: str) -> MCPToolCallResult:
        raise NotImplementedError

    def search_xiaohongshu_feeds(self, *, keyword: str, filters: dict[str, Any] | None = None) -> MCPToolCallResult:
        raise NotImplementedError

    def get_xiaohongshu_feed_detail(self, **arguments: Any) -> MCPToolCallResult:
        raise NotImplementedError


class ContentSourceMCPClient:
    """Adapter used by Agent tools for external content-source MCP tools.

    WeChat article reading can run locally for public article URLs. Xiaohongshu
    read tools require a configured MCP server/login state, so this adapter
    delegates to MCP Gateway when available and returns structured setup errors
    otherwise.
    """

    def __init__(
        self,
        *,
        mcp_client: MCPGatewayClientProtocol | None = None,
        xiaohongshu_base_url: str | None = None,
        xiaohongshu_auth_token: str | None = None,
        http_client: Any | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._mcp_client = mcp_client
        self._xiaohongshu_base_url = xiaohongshu_base_url.rstrip("/") if xiaohongshu_base_url else None
        self._xiaohongshu_auth_token = xiaohongshu_auth_token
        self._http_client = http_client or httpx
        self._timeout_seconds = timeout_seconds

    def read_weixin_article(self, *, url: str) -> MCPToolCallResult:
        parsed = urlparse(url)
        if parsed.netloc != "mp.weixin.qq.com":
            return MCPToolCallResult(
                tool_name="weixin-articles-mcp.read_article",
                ok=False,
                error="WEIXIN_ARTICLE_URL_REQUIRED",
                result={"message": "Expected a public WeChat article URL under mp.weixin.qq.com."},
            )
        try:
            response = self._http_client.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                    )
                },
                follow_redirects=True,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return MCPToolCallResult(
                tool_name="weixin-articles-mcp.read_article",
                ok=False,
                error="WEIXIN_ARTICLE_FETCH_FAILED",
                result={"message": str(exc), "url": url},
            )

        extracted = _extract_weixin_article(response.text)
        return MCPToolCallResult(
            tool_name="weixin-articles-mcp.read_article",
            ok=True,
            result={"url": str(response.url), **extracted},
            metadata={"status_code": response.status_code, "adapter": "local_http"},
        )

    def search_xiaohongshu_feeds(self, *, keyword: str, filters: dict[str, Any] | None = None) -> MCPToolCallResult:
        if self._xiaohongshu_base_url and self._mcp_client is None:
            return self._call_xiaohongshu_rest(
                "xiaohongshu-mcp.search_feeds",
                "/api/v1/feeds/search",
                {"keyword": keyword, "filters": filters},
            )
        return self._call_xiaohongshu_tool(
            "xiaohongshu-mcp.search_feeds",
            {"keyword": keyword, "filters": filters},
        )

    def get_xiaohongshu_feed_detail(self, **arguments: Any) -> MCPToolCallResult:
        if self._xiaohongshu_base_url and self._mcp_client is None:
            payload = dict(arguments)
            if "include_comments" in payload:
                payload["load_all_comments"] = payload.pop("include_comments")
            if "comment_limit" in payload:
                payload["limit"] = payload.pop("comment_limit")
            return self._call_xiaohongshu_rest(
                "xiaohongshu-mcp.get_feed_detail",
                "/api/v1/feeds/detail",
                payload,
            )
        return self._call_xiaohongshu_tool("xiaohongshu-mcp.get_feed_detail", arguments)

    def _call_xiaohongshu_rest(self, tool_name: str, path: str, payload: dict[str, Any]) -> MCPToolCallResult:
        if self._xiaohongshu_base_url is None:
            return self._xiaohongshu_not_configured(tool_name)

        headers = {}
        if self._xiaohongshu_auth_token:
            headers["Authorization"] = f"Bearer {self._xiaohongshu_auth_token}"

        try:
            response = self._http_client.post(
                f"{self._xiaohongshu_base_url}{path}",
                json=payload,
                headers=headers,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            response_payload = response.json()
        except httpx.HTTPError as exc:
            return MCPToolCallResult(
                tool_name=tool_name,
                ok=False,
                error="XIAOHONGSHU_REST_CALL_FAILED",
                result={"message": str(exc), "path": path, "payload": payload},
                metadata={"adapter": "xiaohongshu_rest", "retryable": True},
            )
        except ValueError as exc:
            return MCPToolCallResult(
                tool_name=tool_name,
                ok=False,
                error="XIAOHONGSHU_REST_INVALID_JSON",
                result={"message": str(exc), "path": path},
                metadata={"adapter": "xiaohongshu_rest", "retryable": False},
            )

        success = bool(response_payload.get("success", True)) if isinstance(response_payload, dict) else True
        result = response_payload.get("data", response_payload) if isinstance(response_payload, dict) else response_payload
        message = response_payload.get("message") if isinstance(response_payload, dict) else None
        return MCPToolCallResult(
            tool_name=tool_name,
            ok=success,
            error=None if success else str(message or "XIAOHONGSHU_REST_TOOL_FAILED"),
            result=result,
            metadata={"adapter": "xiaohongshu_rest", "status_code": getattr(response, "status_code", None)},
        )

    def _call_xiaohongshu_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolCallResult:
        if self._mcp_client is None:
            return self._xiaohongshu_not_configured(tool_name)
        return self._mcp_client.call_tool(tool_name=tool_name, arguments=arguments)

    def _xiaohongshu_not_configured(self, tool_name: str) -> MCPToolCallResult:
        return MCPToolCallResult(
            tool_name=tool_name,
            ok=False,
            error="XIAOHONGSHU_MCP_NOT_CONFIGURED",
            result={
                "message": "Xiaohongshu MCP or REST service is not configured. Configure MCP Gateway or Xiaohongshu base URL before using this tool.",
                "tool_name": tool_name,
            },
            metadata={"retryable": False},
        )


def _extract_weixin_article(html: str) -> dict[str, Any]:
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return {"title": "", "author": "", "content_text": html[:10000], "content_html": html}

    soup = BeautifulSoup(html, "html.parser")
    title_node = soup.select_one("#activity-name") or soup.find("title")
    author_node = soup.select_one("#js_name")
    content_node = soup.select_one("#js_content")
    images = [img.get("data-src") or img.get("src") for img in soup.select("#js_content img")]
    return {
        "title": title_node.get_text(" ", strip=True) if title_node else "",
        "author": author_node.get_text(" ", strip=True) if author_node else "",
        "content_text": content_node.get_text("\n", strip=True) if content_node else soup.get_text("\n", strip=True)[:20000],
        "content_html": str(content_node) if content_node else "",
        "images": [image for image in images if image],
    }
