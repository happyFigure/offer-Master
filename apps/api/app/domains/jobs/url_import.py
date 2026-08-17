from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.domains.jobs.models import JobSourceFetchMode, JobSourceType


TRACKING_QUERY_KEYS = {"spm", "from", "share"}
JOB_BOARD_DOMAINS = {
    "zhipin.com",
    "kanzhun.com",
    "51job.com",
    "zhaopin.com",
    "liepin.com",
    "lagou.com",
}
OFFICIAL_CAREER_HOST_LABELS = {"career", "careers", "campus", "join", "job", "jobs"}
OFFICIAL_CAREER_PATH_PREFIXES = ("/career", "/careers", "/campus", "/join", "/jobs")


@dataclass(frozen=True)
class NormalizedImportUrl:
    input_url: str
    normalized_url: str
    normalized_url_hash: str
    domain: str
    removed_query_keys: list[str]
    preserved_query_keys: list[str]


@dataclass(frozen=True)
class UrlImportAnalysis:
    input_url: str
    normalized_url: str
    normalized_url_hash: str
    domain: str
    source_type: JobSourceType
    fetch_mode: JobSourceFetchMode
    fetch_layer: str
    requires_mcp_visible_page: bool
    requires_user_confirmation: bool
    removed_query_keys: list[str]
    preserved_query_keys: list[str]


def normalize_import_url(url: str) -> NormalizedImportUrl:
    input_url = url.strip()
    parsed = urlsplit(input_url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("URL must start with http:// or https://")
    if not parsed.hostname:
        raise ValueError("URL must include a domain")

    domain = parsed.hostname.lower()
    netloc = _normalized_netloc(domain, parsed.port, scheme)
    kept_pairs: list[tuple[str, str]] = []
    removed_keys: set[str] = set()
    preserved_keys: set[str] = set()

    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lowered_key = key.lower()
        if lowered_key.startswith("utm_") or lowered_key in TRACKING_QUERY_KEYS:
            removed_keys.add(key)
            continue
        preserved_keys.add(key)
        kept_pairs.append((key, value))

    normalized_query = urlencode(sorted(kept_pairs, key=lambda item: (item[0].lower(), item[1])))
    normalized_url = urlunsplit((scheme, netloc, parsed.path or "", normalized_query, ""))
    return NormalizedImportUrl(
        input_url=input_url,
        normalized_url=normalized_url,
        normalized_url_hash=sha256(normalized_url.encode("utf-8")).hexdigest(),
        domain=domain,
        removed_query_keys=sorted(removed_keys),
        preserved_query_keys=sorted(preserved_keys),
    )


def analyze_import_url(url: str, source_hint: JobSourceType | None = None) -> UrlImportAnalysis:
    normalized = normalize_import_url(url)
    source_type, fetch_mode, fetch_layer = _classify_source(
        normalized.domain,
        urlsplit(normalized.normalized_url).path,
        source_hint,
    )
    requires_mcp = fetch_mode == JobSourceFetchMode.MCP_VISIBLE_PAGE
    return UrlImportAnalysis(
        input_url=normalized.input_url,
        normalized_url=normalized.normalized_url,
        normalized_url_hash=normalized.normalized_url_hash,
        domain=normalized.domain,
        source_type=source_type,
        fetch_mode=fetch_mode,
        fetch_layer=fetch_layer,
        requires_mcp_visible_page=requires_mcp,
        requires_user_confirmation=requires_mcp,
        removed_query_keys=normalized.removed_query_keys,
        preserved_query_keys=normalized.preserved_query_keys,
    )


def _normalized_netloc(domain: str, port: int | None, scheme: str) -> str:
    if port is None or (scheme == "https" and port == 443) or (scheme == "http" and port == 80):
        return domain
    return f"{domain}:{port}"


def _classify_source(
    domain: str,
    path: str,
    source_hint: JobSourceType | None,
) -> tuple[JobSourceType, JobSourceFetchMode, str]:
    if domain == "mp.weixin.qq.com":
        return JobSourceType.WECHAT_ARTICLE, JobSourceFetchMode.PUBLIC_HTML, "wechat_article"
    if _domain_matches(domain, {"xiaohongshu.com", "xhslink.com"}):
        return JobSourceType.XIAOHONGSHU_NOTE, JobSourceFetchMode.MCP_VISIBLE_PAGE, "mcp_visible_page"
    if _domain_matches(domain, JOB_BOARD_DOMAINS):
        return JobSourceType.JOB_BOARD_VISIBLE_PAGE, JobSourceFetchMode.MCP_VISIBLE_PAGE, "mcp_visible_page"
    if domain.endswith(".edu.cn") or domain.endswith(".edu"):
        return JobSourceType.UNIVERSITY_CAREER_SITE, JobSourceFetchMode.PUBLIC_HTML, "university_career_site"
    if _looks_like_official_career_site(domain, path):
        return JobSourceType.OFFICIAL_CAREER_SITE, JobSourceFetchMode.PUBLIC_HTML, "official_career_site"
    if source_hint is not None:
        return source_hint, _fetch_mode_for_hint(source_hint), _fetch_layer_for_hint(source_hint)
    return JobSourceType.PUBLIC_ARTICLE, JobSourceFetchMode.PUBLIC_HTML, "http_article"


def _domain_matches(domain: str, roots: set[str]) -> bool:
    return any(domain == root or domain.endswith(f".{root}") for root in roots)


def _looks_like_official_career_site(domain: str, path: str) -> bool:
    host_labels = set(domain.split("."))
    return bool(host_labels & OFFICIAL_CAREER_HOST_LABELS) or path.lower().startswith(
        OFFICIAL_CAREER_PATH_PREFIXES
    )


def _fetch_mode_for_hint(source_hint: JobSourceType) -> JobSourceFetchMode:
    if source_hint in {JobSourceType.XIAOHONGSHU_NOTE, JobSourceType.JOB_BOARD_VISIBLE_PAGE}:
        return JobSourceFetchMode.MCP_VISIBLE_PAGE
    if source_hint == JobSourceType.OFFICIAL_API:
        return JobSourceFetchMode.OFFICIAL_API
    return JobSourceFetchMode.PUBLIC_HTML


def _fetch_layer_for_hint(source_hint: JobSourceType) -> str:
    if source_hint == JobSourceType.WECHAT_ARTICLE:
        return "wechat_article"
    if source_hint in {JobSourceType.XIAOHONGSHU_NOTE, JobSourceType.JOB_BOARD_VISIBLE_PAGE}:
        return "mcp_visible_page"
    if source_hint == JobSourceType.UNIVERSITY_CAREER_SITE:
        return "university_career_site"
    if source_hint == JobSourceType.OFFICIAL_CAREER_SITE:
        return "official_career_site"
    if source_hint == JobSourceType.OFFICIAL_API:
        return "official_api"
    return "http_article"
