from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin

import httpx

from app.domains.jobs.providers.base import ProviderStatus


@dataclass(frozen=True)
class UniversityCareerEntry:
    title: str
    source_url: str
    raw_content: str
    raw_payload: dict[str, Any] | None = None


class UniversityCareerProvider:
    name = "university_career"
    source_type = "university_career_site"

    _target_terms = (
        "招聘",
        "校招",
        "校园",
        "宣讲",
        "实习",
        "应届",
        "campus",
        "recruit",
        "recruiting",
        "intern",
        "java",
        "backend",
        "engineer",
        "developer",
        "ai",
        "agent",
        "llm",
        "rag",
    )

    def __init__(self, client: httpx.Client | None = None, timeout_seconds: float = 15.0) -> None:
        self._client = client or httpx.Client()
        self._timeout_seconds = timeout_seconds

    def health_check(self) -> ProviderStatus:
        return ProviderStatus(name=self.name, available=True)

    def fetch(self, entry_url: str, limit: int = 20) -> list[UniversityCareerEntry]:
        response = self._client.get(entry_url, timeout=self._timeout_seconds)
        response.raise_for_status()
        list_html = response.text
        entries: list[UniversityCareerEntry] = []
        seen_urls: set[str] = set()

        for anchor in _extract_anchors(list_html):
            title = _clean_text(anchor.text)
            if title is None or not self._is_recruiting_title(title) or not _is_detail_href(anchor.href):
                continue
            detail_url = urljoin(entry_url, anchor.href)
            if detail_url in seen_urls:
                continue
            seen_urls.add(detail_url)
            detail_text = self._fetch_detail_text(detail_url)
            entries.append(
                UniversityCareerEntry(
                    title=title,
                    source_url=detail_url,
                    raw_content="\n".join(part for part in (title, detail_text) if part),
                    raw_payload={"entry_url": entry_url, "anchor_text": title},
                )
            )
            if len(entries) >= limit:
                break

        return entries

    def _fetch_detail_text(self, detail_url: str) -> str | None:
        try:
            response = self._client.get(detail_url, timeout=self._timeout_seconds)
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        return _clean_text(_extract_text(response.text))

    def _is_recruiting_title(self, title: str) -> bool:
        lowered = title.lower()
        return any(term in title or term in lowered for term in self._target_terms)


@dataclass(frozen=True)
class _Anchor:
    href: str
    text: str


class _AnchorExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.anchors: list[_Anchor] = []
        self._href_stack: list[str] = []
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self._href_stack.append(href)
            self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._href_stack:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._href_stack:
            return
        href = self._href_stack.pop()
        text = _clean_text(" ".join(self._text_parts))
        self._text_parts = []
        if text:
            self.anchors.append(_Anchor(href=href, text=text))


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def _extract_anchors(html: str) -> list[_Anchor]:
    parser = _AnchorExtractor()
    parser.feed(html)
    return parser.anchors


def _extract_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return " ".join(parser.parts)


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(unescape(value).split())
    return cleaned or None


def _is_detail_href(href: str) -> bool:
    lowered = href.lower()
    if lowered in {"/campus", "/jobfair", "/teachin", "/job/search"}:
        return False
    if "job/search" in lowered:
        return False
    detail_markers = ("/view/", "/detail", "id/", "?id=", "/job/")
    return any(marker in lowered for marker in detail_markers)
