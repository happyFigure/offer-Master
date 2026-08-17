from __future__ import annotations

import base64
import re
import zlib
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

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
        list_html = _expand_runtime_injected_html(response.text)
        entries: list[UniversityCareerEntry] = []
        seen_urls: set[str] = set()

        for anchor in _extract_listing_links(list_html):
            title = _clean_text(anchor.text)
            if title is None or not _is_detail_href(anchor.href):
                continue
            if not self._is_recruiting_title(title) and not _is_campus_announcement_href(anchor.href):
                continue
            detail_url = urljoin(entry_url, anchor.href)
            if detail_url in seen_urls:
                continue
            seen_urls.add(detail_url)
            detail_text = self._fetch_detail_text(detail_url)
            raw_payload: dict[str, Any] = {"entry_url": entry_url, "anchor_text": title}
            if anchor.published_at_text:
                raw_payload["published_at_text"] = anchor.published_at_text
            entries.append(
                UniversityCareerEntry(
                    title=title,
                    source_url=detail_url,
                    raw_content="\n".join(part for part in (title, detail_text) if part),
                    raw_payload=raw_payload,
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
    published_at_text: str | None = None


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


def _extract_listing_links(html: str) -> list[_Anchor]:
    structured_links = _extract_info_list_links(html)
    if structured_links:
        return structured_links
    return _extract_anchors(html)


def _extract_info_list_links(html: str) -> list[_Anchor]:
    soup = BeautifulSoup(html, "lxml")
    anchors: list[_Anchor] = []
    for item in soup.select("ul.infoList"):
        anchor = item.find("a", href=True)
        if anchor is None:
            continue
        title = _clean_text(anchor.get_text(" "))
        if not title:
            continue
        date_text = None
        for li in item.find_all("li"):
            if li.find("a", href=True) is not None:
                continue
            candidate_date = _clean_text(li.get_text(" "))
            if candidate_date:
                date_text = candidate_date
                break
        anchors.append(_Anchor(href=str(anchor["href"]), text=title, published_at_text=date_text))
    return anchors


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


def _is_campus_announcement_href(href: str) -> bool:
    lowered = href.lower()
    return "/campus/view/" in lowered or "/campus/detail" in lowered


def _expand_runtime_injected_html(html: str) -> str:
    fragments = []
    pattern = re.compile(
        r"Base64\.decode\(\s*unzip\(\s*\"(?P<payload>[^\"]+)\"\s*\)\.substr\((?P<compressed_offset>\d+)\)\s*\)\.substr\((?P<html_offset>\d+)\)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(html):
        fragment = _decode_base64_zlib_html_fragment(
            match.group("payload"),
            int(match.group("compressed_offset")),
            int(match.group("html_offset")),
        )
        if fragment:
            fragments.append(fragment)
    if not fragments:
        return html
    return "\n".join([*fragments, html])


def _decode_base64_zlib_html_fragment(payload: str, compressed_offset: int, html_offset: int) -> str | None:
    try:
        compressed = base64.b64decode(payload)
    except ValueError:
        return None

    inflated_text = None
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS, zlib.MAX_WBITS | 32):
        try:
            inflated_text = zlib.decompress(compressed, wbits).decode("utf-8")
            break
        except (zlib.error, UnicodeDecodeError):
            continue
    if inflated_text is None or len(inflated_text) <= compressed_offset:
        return None

    encoded_html = inflated_text[compressed_offset:]
    try:
        decoded_html = base64.b64decode(encoded_html).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    return decoded_html[html_offset:] if len(decoded_html) > html_offset else decoded_html
