from __future__ import annotations

from typing import Any

import httpx


def exception_error_details(
    *,
    category: str,
    exc: BaseException,
    url: str | None = None,
    final_url: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "category": category,
        "exception_type": type(exc).__name__,
    }
    message = str(exc)
    if message:
        details["exception_message"] = message[:500]
    if url is not None:
        details["url"] = url
    if final_url is not None:
        details["final_url"] = final_url
    if extra:
        details.update(extra)
    return details


def http_status_error_details(
    exc: httpx.HTTPStatusError,
    *,
    url: str | None = None,
) -> dict[str, Any]:
    response = exc.response
    status_code = response.status_code
    return {
        "category": "http_status",
        "exception_type": type(exc).__name__,
        "url": url or str(response.request.url),
        "final_url": str(response.url),
        "status_code": status_code,
        "status_family": _status_family(status_code),
        "content_type": response.headers.get("content-type", ""),
        "response_preview": _preview(response.text),
    }


def content_quality_error_details(
    *,
    category: str,
    content_length: int,
    text: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "category": category,
        "content_length": content_length,
    }
    if text is not None:
        details["text_preview"] = _preview(text)
    if extra:
        details.update(extra)
    return details


def access_restriction_error_details(
    *,
    detected_marker: str,
    text: str,
    url: str,
    final_url: str | None = None,
    status_code: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "category": "access_restricted",
        "detected_marker": detected_marker,
        "url": url,
        "text_preview": _preview(text),
    }
    if final_url is not None:
        details["final_url"] = final_url
    if status_code is not None:
        details["status_code"] = status_code
        details["status_family"] = _status_family(status_code)
    if extra:
        details.update(extra)
    return details


def _status_family(status_code: int) -> str:
    return f"{status_code // 100}xx"


def _preview(value: str | None, limit: int = 500) -> str:
    if value is None:
        return ""
    return " ".join(value.split())[:limit]
