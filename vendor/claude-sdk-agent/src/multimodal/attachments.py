from __future__ import annotations

import asyncio
import base64
import csv
from dataclasses import dataclass
import io
import mimetypes
from pathlib import Path
import re
from typing import Any, Awaitable, Callable, Mapping, Sequence
from urllib.parse import urlparse

import httpx
from openpyxl import load_workbook


AttachmentFetcher = Callable[[str, Mapping[str, str]], Awaitable[tuple[bytes, str | None]]]

_DATA_URL_PATTERN = re.compile(r"^data:(?P<mime>[-\w.+/]+)?(?:;charset=[^;,]+)?;base64,(?P<data>.+)$", re.IGNORECASE | re.DOTALL)
_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".yaml", ".yml", ".log", ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".xml"}
_CSV_EXTENSIONS = {".csv", ".tsv"}
_SPREADSHEET_EXTENSIONS = {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"}
_ATTACHMENT_HEADER_ALLOWLIST = {
    "accept",
    "accept-language",
    "api-key",
    "authorization",
    "cookie",
    "uac-user-id",
    "uac-user-token",
    "x-api-key",
    "x-uac-user-id",
    "x-uac-user-token",
    "x-user-id",
}
_MAX_SHEETS = 4
_MAX_ROWS_PER_SHEET = 80
_MAX_COLS_PER_ROW = 24
_MAX_CELL_CHARS = 200


@dataclass(slots=True)
class AttachmentRef:
    name: str
    url: str
    mime_type: str
    kind: str
    file_id: str

    @property
    def extension(self) -> str:
        return Path(self.name).suffix.lower()

    @property
    def fingerprint(self) -> str:
        if self.url:
            return f"url:{self.url}"
        if self.file_id:
            return f"file:{self.file_id}"
        return f"name:{self.name}:{self.mime_type}"


def extract_latest_user_parts(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    for item in reversed(messages):
        if not isinstance(item, Mapping):
            continue
        if str(item.get("role") or "").strip().lower() != "user":
            continue
        content = item.get("content")
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
            return [part for part in content if isinstance(part, Mapping)]
        return []
    return []


def extract_metadata_attachments(payload: Mapping[str, Any]) -> list[AttachmentRef]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return []
    items = metadata.get("attachments")
    if not isinstance(items, list):
        return []
    refs: list[AttachmentRef] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        url = str(item.get("downloadUrl") or item.get("url") or "").strip()
        if not url:
            continue
        refs.append(
            AttachmentRef(
                name=str(item.get("name") or "").strip(),
                url=url,
                mime_type=str(item.get("mimeType") or item.get("mime_type") or "").strip().lower(),
                kind=str(item.get("kind") or "").strip().lower(),
                file_id=str(item.get("fileId") or item.get("file_id") or item.get("id") or "").strip(),
            )
        )
    return refs


def part_fingerprint(part: Mapping[str, Any]) -> str:
    image_url = part.get("image_url")
    if isinstance(image_url, Mapping):
        url = str(image_url.get("url") or "").strip()
        if url:
            return f"url:{url}"
    return f"part:{id(part)}"


def build_attachment_download_headers(request_headers: Mapping[str, str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in request_headers.items():
        name = str(key or "").strip()
        if not name:
            continue
        if name.lower() in _ATTACHMENT_HEADER_ALLOWLIST:
            headers[name] = str(value)
    return headers


async def default_attachment_fetcher(url: str, headers: Mapping[str, str], *, timeout_sec: float) -> tuple[bytes, str | None]:
    async with httpx.AsyncClient(timeout=timeout_sec, follow_redirects=True, proxy=None, trust_env=False) as client:
        response = await client.get(url, headers=dict(headers))
        response.raise_for_status()
        return response.content, response.headers.get("content-type")


async def image_part_to_block(
    part: Mapping[str, Any],
    *,
    request_headers: Mapping[str, str],
    timeout_sec: float,
    fetcher: AttachmentFetcher | None = None,
) -> dict[str, Any] | None:
    image_url = part.get("image_url")
    if not isinstance(image_url, Mapping):
        return None
    url = str(image_url.get("url") or "").strip()
    if not url:
        return None
    data, media_type = await _materialize_bytes(
        url,
        fallback_name="image",
        fallback_mime=str(image_url.get("mime_type") or image_url.get("mimeType") or "").strip().lower(),
        request_headers=request_headers,
        timeout_sec=timeout_sec,
        fetcher=fetcher,
    )
    if not media_type.startswith("image/"):
        guessed = _guess_media_type(name="", url=url, fallback=media_type)
        media_type = guessed if guessed.startswith("image/") else "image/png"
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(data).decode("ascii"),
        },
    }


async def attachment_ref_to_blocks(
    ref: AttachmentRef,
    *,
    request_headers: Mapping[str, str],
    timeout_sec: float,
    max_text_chars: int,
    fetcher: AttachmentFetcher | None = None,
) -> list[dict[str, Any]]:
    data, media_type = await _materialize_bytes(
        ref.url,
        fallback_name=ref.name,
        fallback_mime=ref.mime_type,
        request_headers=request_headers,
        timeout_sec=timeout_sec,
        fetcher=fetcher,
    )
    if _is_image_attachment(ref, media_type):
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.b64encode(data).decode("ascii"),
                },
            }
        ]
    if _is_pdf_attachment(ref, media_type):
        return [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.b64encode(data).decode("ascii"),
                },
            }
        ]
    if _is_spreadsheet_attachment(ref, media_type):
        text = await asyncio.to_thread(_spreadsheet_bytes_to_text, data, ref.name, max_text_chars)
        return [_attachment_text_block(ref.name or "spreadsheet", text)]
    if _is_text_attachment(ref, media_type):
        text = await asyncio.to_thread(_text_bytes_to_text, data, ref.name, max_text_chars)
        return [_attachment_text_block(ref.name or "text file", text)]
    return [
        {
            "type": "text",
            "text": (
                f"[Attached file: {ref.name or 'unnamed file'}]\n"
                f"mime_type={media_type or ref.mime_type or 'unknown'}\n"
                "This file format is not parsed directly by the agent yet."
            ),
        }
    ]


async def _materialize_bytes(
    url_or_data: str,
    *,
    fallback_name: str,
    fallback_mime: str,
    request_headers: Mapping[str, str],
    timeout_sec: float,
    fetcher: AttachmentFetcher | None,
) -> tuple[bytes, str]:
    parsed = _parse_data_url(url_or_data)
    if parsed is not None:
        return parsed
    resolved_fetcher = fetcher or (lambda url, headers: default_attachment_fetcher(url, headers, timeout_sec=timeout_sec))
    headers = build_attachment_download_headers(request_headers)
    data, response_mime = await resolved_fetcher(url_or_data, headers)
    media_type = _guess_media_type(name=fallback_name, url=url_or_data, fallback=response_mime or fallback_mime)
    return data, media_type


def _parse_data_url(value: str) -> tuple[bytes, str] | None:
    match = _DATA_URL_PATTERN.match(str(value or "").strip())
    if match is None:
        return None
    media_type = str(match.group("mime") or "application/octet-stream").lower()
    data = base64.b64decode(match.group("data"), validate=False)
    return data, media_type


def _guess_media_type(*, name: str, url: str, fallback: str) -> str:
    fallback_text = str(fallback or "").split(";", 1)[0].strip().lower()
    if fallback_text:
        return fallback_text
    guessed_from_name, _ = mimetypes.guess_type(name or "")
    if guessed_from_name:
        return guessed_from_name.lower()
    guessed_from_url, _ = mimetypes.guess_type(urlparse(url).path)
    if guessed_from_url:
        return guessed_from_url.lower()
    return "application/octet-stream"


def _is_image_attachment(ref: AttachmentRef, media_type: str) -> bool:
    return ref.kind == "image" or media_type.startswith("image/")


def _is_pdf_attachment(ref: AttachmentRef, media_type: str) -> bool:
    return media_type == "application/pdf" or ref.extension == ".pdf"


def _is_spreadsheet_attachment(ref: AttachmentRef, media_type: str) -> bool:
    return (
        ref.extension in _SPREADSHEET_EXTENSIONS
        or media_type in {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
            "application/wps-office.xlsx",
        }
    )


def _is_text_attachment(ref: AttachmentRef, media_type: str) -> bool:
    return (
        media_type.startswith("text/")
        or media_type in {"application/json", "application/xml", "application/yaml"}
        or ref.extension in _TEXT_EXTENSIONS
        or ref.extension in _CSV_EXTENSIONS
    )


def _attachment_text_block(name: str, text: str) -> dict[str, Any]:
    cleaned = text.strip() or "[No readable content extracted]"
    return {
        "type": "text",
        "text": f"[Attached file content: {name}]\n{cleaned}",
    }


def _text_bytes_to_text(data: bytes, name: str, max_text_chars: int) -> str:
    text = _decode_text_bytes(data)
    extension = Path(name).suffix.lower()
    if extension in _CSV_EXTENSIONS:
        return _csv_text_to_text(text, delimiter="\t" if extension == ".tsv" else ",", max_text_chars=max_text_chars)
    return _truncate_text(text, max_text_chars=max_text_chars)


def _spreadsheet_bytes_to_text(data: bytes, name: str, max_text_chars: int) -> str:
    if Path(name).suffix.lower() == ".xls":
        return "Legacy .xls spreadsheets are not parsed directly. Please convert the file to .xlsx."
    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    lines = [f"Spreadsheet: {name or 'unnamed.xlsx'}"]
    truncated = False
    for sheet_index, sheet in enumerate(workbook.worksheets):
        if sheet_index >= _MAX_SHEETS:
            truncated = True
            break
        lines.append(f"\n[Sheet] {sheet.title}")
        row_count = 0
        for row in sheet.iter_rows(values_only=True):
            normalized = [_normalize_cell(value) for value in row[:_MAX_COLS_PER_ROW]]
            if not any(normalized):
                continue
            lines.append("\t".join(normalized))
            row_count += 1
            if row_count >= _MAX_ROWS_PER_SHEET:
                truncated = True
                break
        if row_count == 0:
            lines.append("(empty sheet)")
        if truncated:
            break
    text = "\n".join(lines).strip()
    if truncated:
        text += "\n[Spreadsheet content truncated]"
    return _truncate_text(text, max_text_chars=max_text_chars)


def _csv_text_to_text(text: str, *, delimiter: str, max_text_chars: int) -> str:
    rows: list[str] = []
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    for row_index, row in enumerate(reader):
        if row_index >= _MAX_ROWS_PER_SHEET:
            rows.append("[CSV content truncated]")
            break
        rows.append("\t".join(_normalize_cell(value) for value in row[:_MAX_COLS_PER_ROW]))
    return _truncate_text("\n".join(rows).strip(), max_text_chars=max_text_chars)


def _decode_text_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _normalize_cell(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if len(text) > _MAX_CELL_CHARS:
        return text[:_MAX_CELL_CHARS] + "..."
    return text


def _truncate_text(text: str, *, max_text_chars: int) -> str:
    if len(text) <= max_text_chars:
        return text
    return text[:max_text_chars] + "\n[Content truncated]"
