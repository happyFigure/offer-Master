from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from urllib.parse import urljoin

import httpx
import trafilatura
from bs4 import BeautifulSoup
from readability import Document

from app.domains.jobs.models import DomainHealth
from app.domains.jobs.schemas import ToolErrorCode, ToolResult, ToolSuggestedNextAction
from app.domains.jobs.tool_error_details import (
    content_quality_error_details,
    exception_error_details,
    http_status_error_details,
)
from app.domains.jobs.tool_guard import ToolCallContext, ToolRuntimeGuard


@runtime_checkable
class ContentFetcher(Protocol):
    def fetch(
        self,
        url: str,
        context: ToolCallContext,
        *,
        domain_health: DomainHealth | None = None,
    ) -> ToolResult: ...


@dataclass(frozen=True)
class ExtractedArticleContent:
    title: str | None
    text: str
    candidate_links: list[str]
    extraction_method: str


class HTTPArticleFetcher:
    tool_name = "HTTPArticleFetcher"

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
            response = self._client.get(url, timeout=self._timeout_seconds)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            result = self._failure(
                context,
                ToolErrorCode.FETCH_TIMEOUT,
                str(exc) or "HTTP article fetch timed out",
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
                f"HTTP article fetch failed with status {exc.response.status_code}",
                retryable=500 <= exc.response.status_code < 600,
                next_action=ToolSuggestedNextAction.RETRY_WITH_NEXT_FETCHER,
                artifacts={"url": url, "status_code": exc.response.status_code},
                error_details=http_status_error_details(exc, url=url),
            )
            self._guard.post_record(result, domain_health, now=context.current_time)
            return result
        except httpx.HTTPError as exc:
            result = self._failure(
                context,
                ToolErrorCode.FETCH_FAILED,
                str(exc) or "HTTP article fetch failed",
                retryable=True,
                next_action=ToolSuggestedNextAction.RETRY_WITH_NEXT_FETCHER,
                artifacts={"url": url},
                error_details=exception_error_details(
                    category="network_error",
                    exc=exc,
                    url=url,
                ),
            )
            self._guard.post_record(result, domain_health, now=context.current_time)
            return result

        content_type = response.headers.get("content-type", "")
        article = _extract_article_content(response.text, str(response.url))
        base_artifacts = {
            "url": url,
            "final_url": str(response.url),
            "status_code": response.status_code,
            "content_type": content_type,
            "title": article.title,
            "text": article.text,
            "content_length": len(article.text),
            "candidate_links": article.candidate_links,
            "extraction_method": article.extraction_method,
        }

        if len(article.text) < self._min_text_length:
            result = self._failure(
                context,
                ToolErrorCode.CONTENT_TOO_SHORT,
                "Extracted article content is too short",
                retryable=True,
                next_action=ToolSuggestedNextAction.RETRY_WITH_NEXT_FETCHER,
                artifacts=base_artifacts,
                error_details=content_quality_error_details(
                    category="content_too_short",
                    content_length=len(article.text),
                    text=article.text,
                    extra={
                        "url": url,
                        "final_url": str(response.url),
                        "status_code": response.status_code,
                        "extraction_method": article.extraction_method,
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
            artifacts=base_artifacts,
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
            tool_name=HTTPArticleFetcher.tool_name,
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


def _extract_article_content(html: str, url: str) -> ExtractedArticleContent:
    soup = BeautifulSoup(html, "lxml")
    title = _extract_title(soup)
    candidate_links = _extract_candidate_links(soup, url)

    bs4_text = _extract_with_bs4(soup)
    if bs4_text:
        return ExtractedArticleContent(
            title=title,
            text=bs4_text,
            candidate_links=candidate_links,
            extraction_method="beautifulsoup_lxml",
        )

    trafilatura_text = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
    if trafilatura_text:
        return ExtractedArticleContent(
            title=title,
            text=_clean_text(trafilatura_text),
            candidate_links=candidate_links,
            extraction_method="trafilatura",
        )

    readability_html = Document(html).summary(html_partial=True)
    readability_text = _extract_with_bs4(BeautifulSoup(readability_html, "lxml"))
    return ExtractedArticleContent(
        title=title,
        text=readability_text,
        candidate_links=candidate_links,
        extraction_method="readability_lxml",
    )


def _extract_title(soup: BeautifulSoup) -> str | None:
    heading = soup.find("h1")
    if heading is not None:
        title = _clean_text(heading.get_text(" "))
        if title:
            return title
    if soup.title is not None:
        title = _clean_text(soup.title.get_text(" "))
        if title:
            return title
    return None


def _extract_with_bs4(soup: BeautifulSoup) -> str:
    working = BeautifulSoup(str(soup), "lxml")
    for tag in working(["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    root = working.find("article") or working.find("main") or working.body or working
    return _clean_text(root.get_text(" "))


def _extract_candidate_links(soup: BeautifulSoup, url: str) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href")
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        absolute = urljoin(url, href)
        if absolute in seen:
            continue
        seen.add(absolute)
        links.append(absolute)
    return links


def _clean_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.split())
