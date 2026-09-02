from __future__ import annotations

from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from app.domains.jobs.models import DomainHealth
from app.domains.jobs.schemas import ToolErrorCode, ToolResult, ToolSuggestedNextAction
from app.domains.jobs.tool_error_details import (
    access_restriction_error_details,
    content_quality_error_details,
    exception_error_details,
    http_status_error_details,
)
from app.domains.jobs.tool_guard import ToolCallContext, ToolRuntimeGuard


ACCESS_RESTRICTION_MARKERS = (
    "please open this link in wechat client",
    "captcha",
    "verify",
    "verification",
    "access denied",
    "login required",
    "请在微信客户端打开",
    "验证码",
    "登录",
    "环境异常",
    "访问过于频繁",
)


class WeChatArticleFetcher:
    tool_name = "WeChatArticleFetcher"

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        guard: ToolRuntimeGuard | None = None,
        timeout_seconds: float = 15.0,
        min_text_length: int = 80,
    ) -> None:
        self._client = client or httpx.Client(follow_redirects=True)
        self._guard = guard or ToolRuntimeGuard()
        self._timeout_seconds = timeout_seconds
        self._min_text_length = min_text_length

    def fetch(
        self,
        url: str,
        context: ToolCallContext,
        *,
        domain_health: DomainHealth | None = None,
    ) -> ToolResult:
        guard_result = self._guard.pre_check(context, domain_health=domain_health)
        if not guard_result.ok:
            return guard_result

        try:
            response = self._client.get(
                url,
                timeout=self._timeout_seconds,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                    )
                },
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            result = self._failure(
                context,
                ToolErrorCode.FETCH_TIMEOUT,
                str(exc) or "WeChat article fetch timed out",
                retryable=True,
                next_action=ToolSuggestedNextAction.RETRY_SAME_STAGE,
                artifacts={"url": url},
                error_details=exception_error_details(
                    category="network_timeout",
                    exc=exc,
                    url=url,
                ),
            )
            self._guard.post_record(result, domain_health, now=context.current_time)
            return result
        except httpx.HTTPStatusError as exc:
            result = self._failure(
                context,
                ToolErrorCode.FETCH_FAILED,
                f"WeChat article fetch failed with status {exc.response.status_code}",
                retryable=500 <= exc.response.status_code < 600,
                next_action=ToolSuggestedNextAction.REQUEST_USER_VISIBLE_PAGE,
                artifacts={"url": url, "status_code": exc.response.status_code},
                error_details=http_status_error_details(exc, url=url),
            )
            self._guard.post_record(result, domain_health, now=context.current_time)
            return result
        except httpx.HTTPError as exc:
            result = self._failure(
                context,
                ToolErrorCode.FETCH_FAILED,
                str(exc) or "WeChat article fetch failed",
                retryable=True,
                next_action=ToolSuggestedNextAction.REQUEST_USER_VISIBLE_PAGE,
                artifacts={"url": url},
                error_details=exception_error_details(
                    category="network_error",
                    exc=exc,
                    url=url,
                ),
            )
            self._guard.post_record(result, domain_health, now=context.current_time)
            return result

        html = response.text
        page_text = _clean_text(BeautifulSoup(html, "lxml").get_text(" "))
        detected_marker = _detect_access_restriction_marker(page_text)
        if detected_marker is not None:
            result = self._failure(
                context,
                ToolErrorCode.REQUIRES_MCP_VISIBLE_PAGE,
                "WeChat article requires user-visible page access",
                retryable=True,
                next_action=ToolSuggestedNextAction.REQUEST_USER_VISIBLE_PAGE,
                artifacts={
                    "url": url,
                    "final_url": str(response.url),
                    "status_code": response.status_code,
                    "text_preview": page_text[:200],
                },
                error_details=access_restriction_error_details(
                    detected_marker=detected_marker,
                    text=page_text,
                    url=url,
                    final_url=str(response.url),
                    status_code=response.status_code,
                    extra={"content_type": response.headers.get("content-type", "")},
                ),
            )
            self._guard.post_record(result, domain_health, now=context.current_time)
            return result

        article = _extract_wechat_article(html, str(response.url))
        artifacts = {
            "url": url,
            "final_url": str(response.url),
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "title": article["title"],
            "author": article["author"],
            "published_at": article["published_at"],
            "text": article["text"],
            "content_length": len(article["text"]),
            "candidate_links": article["candidate_links"],
            "extraction_method": "wechat_html",
        }
        if len(article["text"]) < self._min_text_length:
            result = self._failure(
                context,
                ToolErrorCode.CONTENT_TOO_SHORT,
                "WeChat article content is empty or too short",
                retryable=True,
                next_action=ToolSuggestedNextAction.REQUEST_MANUAL_PASTE,
                artifacts=artifacts,
                error_details=content_quality_error_details(
                    category="content_too_short",
                    content_length=len(article["text"]),
                    text=str(article["text"]),
                    extra={
                        "url": url,
                        "final_url": str(response.url),
                        "status_code": response.status_code,
                        "extraction_method": "wechat_html",
                    },
                ),
            )
            self._guard.post_record(result, domain_health, now=context.current_time)
            return result

        result = ToolResult(
            ok=True,
            stage=context.stage,
            tool_name=self.tool_name,
            retryable=False,
            suggested_next_action=ToolSuggestedNextAction.CONTINUE_WORKFLOW,
            cost={
                "tool_calls": context.tool_call_count + 1,
                "llm_calls": context.llm_call_count,
                "fetch_attempts_for_stage": context.fetch_attempts_for_stage + 1,
                "mcp_requests": context.mcp_request_count,
            },
            artifacts=artifacts,
        )
        self._guard.post_record(result, domain_health, now=context.current_time)
        return result

    @staticmethod
    def _failure(
        context: ToolCallContext,
        error_code: ToolErrorCode,
        error_message: str,
        *,
        retryable: bool,
        next_action: ToolSuggestedNextAction,
        artifacts: dict[str, object],
        error_details: dict[str, object] | None = None,
    ) -> ToolResult:
        return ToolResult(
            ok=False,
            stage=context.stage,
            tool_name=WeChatArticleFetcher.tool_name,
            error_code=error_code,
            error_message=error_message,
            retryable=retryable,
            suggested_next_action=next_action,
            error_details=error_details or {},
            cost={
                "tool_calls": context.tool_call_count + 1,
                "llm_calls": context.llm_call_count,
                "fetch_attempts_for_stage": context.fetch_attempts_for_stage + 1,
                "mcp_requests": context.mcp_request_count,
            },
            artifacts=artifacts,
        )


def _extract_wechat_article(html: str, url: str) -> dict[str, object]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    content_root = soup.select_one("#js_content") or soup.select_one("#img-content") or soup.body or soup
    return {
        "title": _extract_title(soup),
        "author": _text_from_selector(soup, "#js_name")
        or _text_from_selector(soup, ".rich_media_meta_text")
        or _meta_content(soup, "name", "author"),
        "published_at": _text_from_selector(soup, "#publish_time")
        or _text_from_selector(soup, "#js_publish_time")
        or _meta_content(soup, "property", "article:published_time"),
        "text": _clean_text(content_root.get_text(" ")),
        "candidate_links": _extract_candidate_links(content_root, url),
    }


def _extract_title(soup: BeautifulSoup) -> str | None:
    return (
        _text_from_selector(soup, "#activity-name")
        or _meta_content(soup, "property", "og:title")
        or _text_from_selector(soup, "h1")
        or (soup.title and _clean_text(soup.title.get_text(" ")))
        or None
    )


def _text_from_selector(soup: BeautifulSoup, selector: str) -> str | None:
    element = soup.select_one(selector)
    if element is None:
        return None
    text = _clean_text(element.get_text(" "))
    return text or None


def _meta_content(soup: BeautifulSoup, key: str, value: str) -> str | None:
    element = soup.find("meta", attrs={key: value})
    if element is None:
        return None
    content = element.get("content")
    if not isinstance(content, str):
        return None
    text = _clean_text(content)
    return text or None


def _extract_candidate_links(root: BeautifulSoup, url: str) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for anchor in root.find_all("a", href=True):
        href = anchor.get("href")
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        absolute = urljoin(url, href)
        if absolute in seen:
            continue
        seen.add(absolute)
        links.append(absolute)
    return links


def _looks_access_restricted(text: str) -> bool:
    return _detect_access_restriction_marker(text) is not None


def _detect_access_restriction_marker(text: str) -> str | None:
    lowered = text.lower()
    for marker in ACCESS_RESTRICTION_MARKERS:
        if marker in lowered:
            return marker
    return None


def _clean_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.split())
