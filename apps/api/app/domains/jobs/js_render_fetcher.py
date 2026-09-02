from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import urljoin

import httpx
import trafilatura
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
    "login required",
    "please login",
    "sign in",
    "captcha",
    "verify captcha",
    "access denied",
    "forbidden",
    "too many requests",
    "\u767b\u5f55",
    "\u8bf7\u767b\u5f55",
    "\u9a8c\u8bc1",
    "\u9a8c\u8bc1\u7801",
    "\u8bbf\u95ee\u53d7\u9650",
    "\u64cd\u4f5c\u8fc7\u4e8e\u9891\u7e41",
)


@dataclass(frozen=True)
class PlaywrightRuntimePaths:
    browsers_path: Path
    user_data_dir: Path
    temp_dir: Path

    def ensure(self) -> None:
        for path in (self.browsers_path, self.user_data_dir, self.temp_dir):
            path.mkdir(parents=True, exist_ok=True)

    def environment(self) -> dict[str, str]:
        return {
            "PLAYWRIGHT_BROWSERS_PATH": str(self.browsers_path),
            "TEMP": str(self.temp_dir),
            "TMP": str(self.temp_dir),
        }


@dataclass(frozen=True)
class RenderedPage:
    final_url: str
    status_code: int | None
    title: str | None
    html: str
    visible_text: str
    candidate_links: list[str]


@runtime_checkable
class PageRenderer(Protocol):
    def render(self, url: str) -> RenderedPage: ...


@dataclass(frozen=True)
class Crawl4AIExtractedContent:
    title: str | None
    markdown: str
    text: str
    candidate_links: list[str]
    extraction_method: str


@dataclass(frozen=True)
class Crawl4AIPageResult:
    final_url: str
    status_code: int | None
    content_type: str
    content: Crawl4AIExtractedContent


@runtime_checkable
class Crawl4AIExtractor(Protocol):
    def extract(self, *, html: str, url: str) -> Crawl4AIExtractedContent: ...


class PlaywrightPageRenderer:
    def __init__(
        self,
        *,
        runtime_paths: PlaywrightRuntimePaths | None = None,
        timeout_ms: int = 30_000,
    ) -> None:
        self._runtime_paths = runtime_paths or build_playwright_runtime_paths()
        self._timeout_ms = timeout_ms

    def render(self, url: str) -> RenderedPage:
        self._runtime_paths.ensure()
        previous_env = {key: os.environ.get(key) for key in self._runtime_paths.environment()}
        os.environ.update(self._runtime_paths.environment())
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as playwright:
                context = playwright.chromium.launch_persistent_context(
                    user_data_dir=str(self._runtime_paths.user_data_dir),
                    headless=True,
                    accept_downloads=False,
                    viewport={"width": 1365, "height": 900},
                    ignore_https_errors=True,
                )
                try:
                    page = context.new_page()
                    response = page.goto(url, wait_until="networkidle", timeout=self._timeout_ms)
                    title = page.title()
                    html = page.content()
                    visible_text = _safe_body_text(page, self._timeout_ms)
                    links = page.eval_on_selector_all(
                        "a[href]",
                        "els => Array.from(new Set(els.map(a => a.href).filter(Boolean)))",
                    )
                    return RenderedPage(
                        final_url=page.url,
                        status_code=response.status if response is not None else None,
                        title=title or None,
                        html=html,
                        visible_text=_clean_text(visible_text),
                        candidate_links=list(links or []),
                    )
                finally:
                    context.close()
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class PlaywrightFetcher:
    tool_name = "PlaywrightFetcher"

    def __init__(
        self,
        *,
        renderer: PageRenderer | None = None,
        guard: ToolRuntimeGuard | None = None,
        min_text_length: int = 80,
    ) -> None:
        self._renderer = renderer or PlaywrightPageRenderer()
        self._guard = guard or ToolRuntimeGuard()
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
            page = self._renderer.render(url)
        except TimeoutError as exc:
            result = self._failure(
                context,
                ToolErrorCode.FETCH_TIMEOUT,
                str(exc) or "Playwright render timed out",
                retryable=True,
                next_action=ToolSuggestedNextAction.RETRY_SAME_STAGE,
                artifacts={"url": url},
                error_details=exception_error_details(
                    category="renderer_timeout",
                    exc=exc,
                    url=url,
                ),
            )
            self._guard.post_record(result, domain_health, now=context.current_time)
            return result
        except Exception as exc:
            result = self._failure(
                context,
                ToolErrorCode.FETCH_FAILED,
                str(exc) or "Playwright render failed",
                retryable=True,
                next_action=ToolSuggestedNextAction.RETRY_WITH_NEXT_FETCHER,
                artifacts={"url": url},
                error_details=exception_error_details(
                    category="renderer_exception",
                    exc=exc,
                    url=url,
                ),
            )
            self._guard.post_record(result, domain_health, now=context.current_time)
            return result

        text = _clean_text(page.visible_text)
        artifacts = {
            "url": url,
            "final_url": page.final_url,
            "status_code": page.status_code,
            "title": page.title,
            "text": text,
            "content_length": len(text),
            "candidate_links": page.candidate_links,
            "html_length": len(page.html or ""),
            "extraction_method": "playwright_visible_text",
        }
        detected_marker = _detect_access_restriction_marker(page.title, text, page.html)
        if detected_marker is not None:
            result = self._failure(
                context,
                ToolErrorCode.REQUIRES_MCP_VISIBLE_PAGE,
                "Rendered page requires login, captcha, or user-visible access",
                retryable=True,
                next_action=ToolSuggestedNextAction.REQUEST_USER_VISIBLE_PAGE,
                artifacts=artifacts,
                error_details=access_restriction_error_details(
                    detected_marker=detected_marker,
                    text=text or page.html,
                    url=url,
                    final_url=page.final_url,
                    status_code=page.status_code,
                    extra={"title": page.title, "html_length": len(page.html or "")},
                ),
            )
            self._guard.post_record(result, domain_health, now=context.current_time)
            return result
        if len(text) < self._min_text_length:
            result = self._failure(
                context,
                ToolErrorCode.CONTENT_TOO_SHORT,
                "Rendered page visible text is too short",
                retryable=True,
                next_action=ToolSuggestedNextAction.RETRY_WITH_NEXT_FETCHER,
                artifacts=artifacts,
                error_details=content_quality_error_details(
                    category="content_too_short",
                    content_length=len(text),
                    text=text,
                    extra={
                        "url": url,
                        "final_url": page.final_url,
                        "status_code": page.status_code,
                        "extraction_method": "playwright_visible_text",
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
            cost=_next_fetch_cost(context),
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
            tool_name=PlaywrightFetcher.tool_name,
            error_code=error_code,
            error_message=error_message,
            retryable=retryable,
            suggested_next_action=next_action,
            error_details=error_details or {},
            cost=_next_fetch_cost(context),
            artifacts=artifacts,
        )


class DefaultCrawl4AIExtractor:
    def extract(self, *, html: str, url: str) -> Crawl4AIExtractedContent:
        soup = BeautifulSoup(html, "lxml")
        markdown = trafilatura.extract(
            html,
            url=url,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
        ) or ""
        text = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
        if not text:
            text = _extract_text_from_html(soup)
        return Crawl4AIExtractedContent(
            title=_extract_title(soup),
            markdown=_clean_text(markdown),
            text=_clean_text(text),
            candidate_links=_extract_candidate_links(soup, url),
            extraction_method="crawl4ai_markdown",
        )


class Crawl4AIUrlCrawler:
    def __init__(
        self,
        *,
        runtime_paths: PlaywrightRuntimePaths | None = None,
        timeout_ms: int = 30_000,
    ) -> None:
        self._runtime_paths = runtime_paths or build_playwright_runtime_paths()
        self._timeout_ms = timeout_ms

    def extract_url(self, url: str) -> Crawl4AIPageResult:
        self._runtime_paths.ensure()
        previous_env = {key: os.environ.get(key) for key in self._runtime_paths.environment()}
        os.environ.update(self._runtime_paths.environment())
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self._extract_url_async(url))
            raise RuntimeError("Crawl4AIUrlCrawler.extract_url cannot run inside an active event loop")
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    async def _extract_url_async(self, url: str) -> Crawl4AIPageResult:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

        browser_config = BrowserConfig(
            headless=True,
            use_persistent_context=True,
            user_data_dir=str(self._runtime_paths.user_data_dir),
            downloads_path=str(self._runtime_paths.temp_dir),
            verbose=False,
            enable_stealth=False,
            magic=False,
            simulate_user=False,
        )
        run_config = CrawlerRunConfig(
            word_count_threshold=1,
            wait_until="networkidle",
            page_timeout=self._timeout_ms,
            remove_forms=True,
            remove_overlay_elements=False,
            simulate_user=False,
            magic=False,
        )
        async with AsyncWebCrawler(
            config=browser_config,
            base_directory=str(self._runtime_paths.temp_dir),
            thread_safe=False,
        ) as crawler:
            result = _first_crawl_result(await crawler.arun(url=url, config=run_config))

        if not getattr(result, "success", False):
            raise RuntimeError(getattr(result, "error_message", None) or "Crawl4AI crawl failed")

        markdown = _crawl_markdown(result)
        html = getattr(result, "cleaned_html", None) or getattr(result, "html", "") or ""
        soup = BeautifulSoup(html, "lxml")
        text = _clean_text(markdown) or _extract_text_from_html(soup)
        links = _crawl_links(result) or _extract_candidate_links(soup, url)
        return Crawl4AIPageResult(
            final_url=getattr(result, "redirected_url", None) or getattr(result, "url", None) or url,
            status_code=getattr(result, "status_code", None),
            content_type=_content_type_from_headers(getattr(result, "response_headers", None)),
            content=Crawl4AIExtractedContent(
                title=_extract_title_from_metadata(getattr(result, "metadata", None)) or _extract_title(soup),
                markdown=markdown,
                text=text,
                candidate_links=links,
                extraction_method="crawl4ai_markdown",
            ),
        )


class Crawl4AIFetcher:
    tool_name = "Crawl4AIFetcher"

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        crawler: Crawl4AIUrlCrawler | None = None,
        extractor: Crawl4AIExtractor | None = None,
        guard: ToolRuntimeGuard | None = None,
        timeout_seconds: float = 15.0,
        min_text_length: int = 80,
    ) -> None:
        self._use_url_crawler = client is None and extractor is None
        self._client = client or httpx.Client(follow_redirects=True)
        self._crawler = crawler or (Crawl4AIUrlCrawler() if self._use_url_crawler else None)
        self._extractor = extractor or DefaultCrawl4AIExtractor()
        self._guard = guard or ToolRuntimeGuard()
        self._timeout_seconds = timeout_seconds
        self._min_text_length = min_text_length

    def fetch(
        self,
        url: str,
        context: ToolCallContext,
        *,
        domain_health: DomainHealth | None = None,
        html: str | None = None,
    ) -> ToolResult:
        guard_result = self._guard.pre_check(context, domain_health=domain_health)
        if not guard_result.ok:
            return guard_result

        if html is None and self._crawler is not None:
            try:
                crawled = self._crawler.extract_url(url)
            except TimeoutError as exc:
                result = self._failure(
                    context,
                    ToolErrorCode.FETCH_TIMEOUT,
                    str(exc) or "Crawl4AI crawl timed out",
                    retryable=True,
                    next_action=ToolSuggestedNextAction.RETRY_SAME_STAGE,
                    artifacts={"url": url},
                    error_details=exception_error_details(
                        category="crawl4ai_timeout",
                        exc=exc,
                        url=url,
                    ),
                )
                self._guard.post_record(result, domain_health, now=context.current_time)
                return result
            except Exception as exc:
                result = self._failure(
                    context,
                    ToolErrorCode.FETCH_FAILED,
                    str(exc) or "Crawl4AI crawl failed",
                    retryable=True,
                    next_action=ToolSuggestedNextAction.REQUEST_MANUAL_PASTE,
                    artifacts={"url": url},
                    error_details=exception_error_details(
                        category="crawl4ai_crawler_exception",
                        exc=exc,
                        url=url,
                    ),
                )
                self._guard.post_record(result, domain_health, now=context.current_time)
                return result
            return self._result_from_content(
                url=url,
                response_url=crawled.final_url,
                status_code=crawled.status_code,
                content_type=crawled.content_type,
                content=crawled.content,
                context=context,
                domain_health=domain_health,
            )

        response_url = url
        status_code: int | None = None
        content_type = ""
        if html is None:
            try:
                response = self._client.get(url, timeout=self._timeout_seconds)
                response.raise_for_status()
            except httpx.TimeoutException as exc:
                result = self._failure(
                    context,
                    ToolErrorCode.FETCH_TIMEOUT,
                    str(exc) or "Crawl4AI fetch timed out",
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
                    f"Crawl4AI fetch failed with status {exc.response.status_code}",
                    retryable=500 <= exc.response.status_code < 600,
                    next_action=ToolSuggestedNextAction.REQUEST_MANUAL_PASTE,
                    artifacts={"url": url, "status_code": exc.response.status_code},
                    error_details=http_status_error_details(exc, url=url),
                )
                self._guard.post_record(result, domain_health, now=context.current_time)
                return result
            except httpx.HTTPError as exc:
                result = self._failure(
                    context,
                    ToolErrorCode.FETCH_FAILED,
                    str(exc) or "Crawl4AI fetch failed",
                    retryable=True,
                    next_action=ToolSuggestedNextAction.REQUEST_MANUAL_PASTE,
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
            response_url = str(response.url)
            status_code = response.status_code
            content_type = response.headers.get("content-type", "")

        try:
            content = self._extractor.extract(html=html, url=response_url)
        except Exception as exc:
            result = self._failure(
                context,
                ToolErrorCode.CONTENT_EXTRACTION_FAILED,
                str(exc) or "Crawl4AI content extraction failed",
                retryable=True,
                next_action=ToolSuggestedNextAction.REQUEST_MANUAL_PASTE,
                artifacts={"url": url, "final_url": response_url, "status_code": status_code},
                error_details=exception_error_details(
                    category="content_extraction_exception",
                    exc=exc,
                    url=url,
                    final_url=response_url,
                    extra={"status_code": status_code},
                ),
            )
            self._guard.post_record(result, domain_health, now=context.current_time)
            return result

        return self._result_from_content(
            url=url,
            response_url=response_url,
            status_code=status_code,
            content_type=content_type,
            content=content,
            context=context,
            domain_health=domain_health,
        )

    def _result_from_content(
        self,
        *,
        url: str,
        response_url: str,
        status_code: int | None,
        content_type: str,
        content: Crawl4AIExtractedContent,
        context: ToolCallContext,
        domain_health: DomainHealth | None,
    ) -> ToolResult:
        text = _clean_text(content.text)
        artifacts = {
            "url": url,
            "final_url": response_url,
            "status_code": status_code,
            "content_type": content_type,
            "title": content.title,
            "markdown": content.markdown,
            "text": text,
            "content_length": len(text),
            "candidate_links": content.candidate_links,
            "extraction_method": content.extraction_method,
        }
        if len(text) < self._min_text_length:
            result = self._failure(
                context,
                ToolErrorCode.CONTENT_TOO_SHORT,
                "Crawl4AI extracted content is too short",
                retryable=True,
                next_action=ToolSuggestedNextAction.REQUEST_MANUAL_PASTE,
                artifacts=artifacts,
                error_details=content_quality_error_details(
                    category="content_too_short",
                    content_length=len(text),
                    text=text,
                    extra={
                        "url": url,
                        "final_url": response_url,
                        "status_code": status_code,
                        "extraction_method": content.extraction_method,
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
            cost=_next_fetch_cost(context),
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
            tool_name=Crawl4AIFetcher.tool_name,
            error_code=error_code,
            error_message=error_message,
            retryable=retryable,
            suggested_next_action=next_action,
            error_details=error_details or {},
            cost=_next_fetch_cost(context),
            artifacts=artifacts,
        )


def build_playwright_runtime_paths(project_root: Path | str | None = None) -> PlaywrightRuntimePaths:
    root = Path(project_root).resolve() if project_root is not None else _default_project_root()
    if _is_c_drive(root):
        raise ValueError("Playwright runtime paths must not be placed on C drive")
    return PlaywrightRuntimePaths(
        browsers_path=root / ".external" / "ms-playwright",
        user_data_dir=root / "runtime" / "playwright" / "user-data",
        temp_dir=root / "runtime" / "temp",
    )


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _is_c_drive(path: Path) -> bool:
    return path.drive.lower() == "c:"


def _safe_body_text(page: object, timeout_ms: int) -> str:
    try:
        return page.locator("body").inner_text(timeout=timeout_ms)
    except Exception:
        return ""


def _looks_access_restricted(title: str | None, text: str, html: str) -> bool:
    return _detect_access_restriction_marker(title, text, html) is not None


def _detect_access_restriction_marker(title: str | None, text: str, html: str) -> str | None:
    combined = " ".join((title or "", text or "", html or "")).lower()
    for marker in ACCESS_RESTRICTION_MARKERS:
        if marker in combined:
            return marker
    return None


def _first_crawl_result(result_container: object) -> object:
    if isinstance(result_container, list) and result_container:
        return result_container[0]
    results = getattr(result_container, "results", None)
    if isinstance(results, list) and results:
        return results[0]
    return result_container


def _crawl_markdown(result: object) -> str:
    markdown_result = getattr(result, "markdown", None) or getattr(result, "markdown_v2", None)
    if isinstance(markdown_result, str):
        return _clean_text(markdown_result)
    for attr in ("fit_markdown", "raw_markdown", "markdown_with_citations"):
        value = getattr(markdown_result, attr, None)
        if isinstance(value, str) and value.strip():
            return _clean_text(value)
    return ""


def _crawl_links(result: object) -> list[str]:
    links = getattr(result, "links", None)
    if not isinstance(links, dict):
        return []
    values: list[str] = []
    seen: set[str] = set()
    for group in links.values():
        if not isinstance(group, list):
            continue
        for item in group:
            href = item.get("href") if isinstance(item, dict) else None
            if not isinstance(href, str) or not href or href in seen:
                continue
            seen.add(href)
            values.append(href)
    return values


def _content_type_from_headers(headers: object) -> str:
    if not isinstance(headers, dict):
        return ""
    for key, value in headers.items():
        if str(key).lower() == "content-type" and isinstance(value, str):
            return value
    return ""


def _extract_title_from_metadata(metadata: object) -> str | None:
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("title") or metadata.get("og:title")
    return _clean_text(value) if isinstance(value, str) and value.strip() else None


def _next_fetch_cost(context: ToolCallContext) -> dict[str, int]:
    return {
        "tool_calls": context.tool_call_count + 1,
        "llm_calls": context.llm_call_count,
        "fetch_attempts_for_stage": context.fetch_attempts_for_stage + 1,
        "mcp_requests": context.mcp_request_count,
    }


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


def _extract_text_from_html(soup: BeautifulSoup) -> str:
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
