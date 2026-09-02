from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import unescape
import re
from urllib.parse import urlencode, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from app.domains.jobs.models import JobSource


RECRUITING_TERMS = (
    "招聘",
    "校招",
    "校园招聘",
    "秋招",
    "春招",
    "实习",
    "提前批",
    "网申",
    "2026",
    "2027",
)

EXCLUDED_TERMS = (
    "求职补贴",
    "就业政策",
    "就业手续",
    "指导员",
    "辅导员",
    "公开招聘",
    "国家大学生就业服务平台",
    "研究生报名",
    "推免",
    "国省考",
    "公务员考试",
    "一轮复习",
    "生肖",
    "油价",
    "铜价",
)


@dataclass(frozen=True)
class SearchWeChatArticleEntry:
    title: str
    url: str
    source_account: str | None = None
    published_at: datetime | None = None
    raw_payload: dict | None = None


class WeChatAccountSearchProvider:
    name = "wechat_account_search"

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        timeout_seconds: float = 15.0,
        search_base_url: str = "https://weixin.sogou.com/weixin",
    ) -> None:
        self._client = client or httpx.Client(follow_redirects=False)
        self._timeout_seconds = timeout_seconds
        self._search_base_url = search_base_url

    def discover(self, source: JobSource, limit: int) -> list[SearchWeChatArticleEntry]:
        entries: list[SearchWeChatArticleEntry] = []
        seen_urls: set[str] = set()
        allowed_accounts = _source_query_bases(source)
        for query in _build_queries(source):
            html = self._search(query)
            for result in _parse_search_results(html):
                if not _is_allowed_source_account(result.source_account, allowed_accounts):
                    continue
                if not _is_recruiting_article_title(result.title):
                    continue
                if _is_outdated_article_title(result.title):
                    continue
                resolved_url = self._resolve_article_url(result.url)
                if not resolved_url or resolved_url in seen_urls:
                    continue
                seen_urls.add(resolved_url)
                entries.append(
                    SearchWeChatArticleEntry(
                        title=result.title,
                        url=resolved_url,
                        source_account=result.source_account or source.name,
                        raw_payload={
                            "discovery_method": "sogou_weixin_search",
                            "search_query": query,
                            "search_url": self._search_url(query),
                            "raw_url": result.url,
                            "published_at_text": result.published_at_text,
                        },
                    )
                )
                if len(entries) >= limit:
                    return entries
        return entries

    def _search(self, query: str) -> str:
        response = self._client.get(
            self._search_url(query),
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
                ),
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        return response.text

    def _search_url(self, query: str) -> str:
        return f"{self._search_base_url}?{urlencode({'type': '2', 'query': query, 'ie': 'utf8'})}"

    def _resolve_article_url(self, url: str) -> str | None:
        absolute_url = urljoin("https://weixin.sogou.com/weixin", url)
        if _is_wechat_article_url(absolute_url):
            return absolute_url
        if urlparse(absolute_url).netloc != "weixin.sogou.com":
            return absolute_url
        try:
            response = self._client.get(
                absolute_url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
                    )
                },
                timeout=self._timeout_seconds,
                follow_redirects=False,
            )
        except httpx.HTTPError:
            return absolute_url
        location = response.headers.get("location") or response.headers.get("Location")
        if not location:
            return absolute_url
        resolved = urljoin(absolute_url, location)
        return resolved if _is_wechat_article_url(resolved) else absolute_url


@dataclass(frozen=True)
class _SearchResult:
    title: str
    url: str
    source_account: str | None = None
    published_at_text: str | None = None


def _build_queries(source: JobSource) -> list[str]:
    base_terms = _source_query_bases(source)
    queries = []
    for base in base_terms:
        cleaned = _clean_text(base)
        if not cleaned:
            continue
        queries.extend(
            [
                f"{cleaned} 招聘 2027 校园招聘",
                f"{cleaned} 校招 秋招",
                f"{cleaned} 实习 招聘",
            ]
        )
    return list(dict.fromkeys(queries))


def _source_query_bases(source: JobSource) -> list[str]:
    base_terms = [source.name]
    if source.entry_url and not source.entry_url.startswith(("http://", "https://")):
        base_terms.append(source.entry_url)

    aliases = []
    for base in base_terms:
        cleaned = _clean_text(base)
        if not cleaned:
            continue
        if cleaned.endswith("公众号"):
            without_suffix = cleaned[: -len("公众号")].strip()
            if without_suffix.endswith("大学"):
                aliases.append(f"{without_suffix[: -len('大学')]}就业")
        aliases.append(cleaned)
        if cleaned.endswith("公众号"):
            without_suffix = cleaned[: -len("公众号")].strip()
            if without_suffix:
                aliases.append(without_suffix)
    return list(dict.fromkeys(aliases))


def _parse_search_results(html: str) -> list[_SearchResult]:
    soup = BeautifulSoup(html, "lxml")
    results: list[_SearchResult] = []
    for item in soup.select("li"):
        title_anchor = item.select_one("h3 a[href]") or item.find("a", href=True)
        if title_anchor is None:
            continue
        title = _clean_text(title_anchor.get_text(" "))
        href = title_anchor.get("href")
        if not title or not href:
            continue
        account_text = _clean_text(_first_text(item, ["a.account", ".account", ".s-p"]))
        published_text = _clean_text(_first_text(item, [".s2", ".time", ".date"]))
        results.append(
            _SearchResult(
                title=title,
                url=str(href),
                source_account=account_text,
                published_at_text=published_text,
            )
        )
    return results


def _first_text(soup, selectors: list[str]) -> str | None:
    for selector in selectors:
        node = soup.select_one(selector)
        if node is not None:
            return node.get_text(" ")
    return None


def _is_recruiting_article_title(title: str) -> bool:
    normalized = _clean_text(title) or ""
    compact = normalized.replace(" ", "")
    if any(term in normalized or term in compact for term in EXCLUDED_TERMS):
        return False
    return any(term in normalized or term in compact for term in RECRUITING_TERMS)


def _is_allowed_source_account(source_account: str | None, allowed_accounts: list[str]) -> bool:
    if not source_account:
        return True
    account = source_account.replace(" ", "")
    return any(base.replace(" ", "") in account or account in base.replace(" ", "") for base in allowed_accounts)


def _is_outdated_article_title(title: str) -> bool:
    years = [int(value) for value in re.findall(r"20\d{2}", title)]
    return bool(years) and max(years) < 2026


def _is_wechat_article_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == "mp.weixin.qq.com" and parsed.path.startswith("/s/")


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(unescape(value).split())
    return cleaned or None
