"""Read-only QQ Mail MCP connector.

The connector deliberately uses only Python's standard library. It opens the
INBOX in read-only mode and fetches messages with BODY.PEEK so an Agent lookup
does not mutate mailbox state.
"""

from __future__ import annotations

from datetime import date, timedelta
from email import policy
from email.header import decode_header, make_header
from email.message import Message
from email.parser import BytesParser
import imaplib
import os
import re
from typing import Any

from mcp.server import MCPServer


IMAP_HOST = "imap.qq.com"
IMAP_PORT = 993
_UID_PATTERN = re.compile(r"^[0-9]{1,32}$")

server = MCPServer(
    "qq-mail-read-only",
    instructions=(
        "Read-only QQ Mail connector for recruitment notifications. "
        "Never send, delete, move, or mark messages as read."
    ),
)


@server.tool()
def list_messages(
    limit: int = 20,
    since_days: int = 30,
    subject: str | None = None,
    sender: str | None = None,
) -> dict[str, Any]:
    """List recent INBOX message headers without marking messages as read."""

    _validate_list_args(limit, since_days, subject, sender)
    with _mailbox() as mailbox:
        criteria = ["SINCE", (date.today() - timedelta(days=since_days)).strftime("%d-%b-%Y")]
        if subject:
            criteria.extend(("SUBJECT", _imap_text(subject)))
        if sender:
            criteria.extend(("FROM", _imap_text(sender)))
        status, data = mailbox.uid("SEARCH", None, *criteria)
        if status != "OK":
            raise RuntimeError("QQ Mail search failed")
        uids = [item for item in (data[0] or b"").decode("ascii", "ignore").split() if _UID_PATTERN.fullmatch(item)]
        selected = list(reversed(uids[-limit:]))
        messages: list[dict[str, Any]] = []
        for uid in selected:
            status, fetched = mailbox.uid(
                "FETCH",
                uid,
                "(BODY.PEEK[HEADER.FIELDS (FROM TO SUBJECT DATE MESSAGE-ID)])",
            )
            if status != "OK":
                continue
            raw = _first_bytes(fetched)
            header = BytesParser(policy=policy.default).parsebytes(raw or b"")
            messages.append(_header_payload(uid, header))
        return {"ok": True, "read_only": True, "folder": "INBOX", "messages": messages}


@server.tool()
def get_message(uid: str, include_body: bool = True, max_chars: int = 12000) -> dict[str, Any]:
    """Read one INBOX message using BODY.PEEK without changing its read flag."""

    _validate_message_args(uid, include_body, max_chars)
    with _mailbox() as mailbox:
        flags_status, flags_data = mailbox.uid("FETCH", uid, "(FLAGS)")
        if flags_status != "OK":
            raise RuntimeError("QQ Mail message flags could not be checked")
        flags_before = _flags_from_fetch(flags_data)
        status, fetched = mailbox.uid("FETCH", uid, "(BODY.PEEK[] FLAGS)")
        if status != "OK":
            raise RuntimeError("QQ Mail message fetch failed")
        raw = _first_bytes(fetched)
        if raw is None:
            raise RuntimeError("QQ Mail returned an empty message")
        message = BytesParser(policy=policy.default).parsebytes(raw)
        payload = _header_payload(uid, message)
        flags_after = _flags_from_fetch(fetched)
        payload["flags_before"] = flags_before
        payload["flags_after"] = flags_after
        payload["seen_state_unchanged"] = ("\\Seen" in flags_before) == ("\\Seen" in flags_after)
        if include_body:
            payload["body"] = _message_body(message, max_chars=max_chars)
        return {"ok": True, "read_only": True, "message": payload}


def _mailbox() -> imaplib.IMAP4_SSL:
    username = os.environ.get("JOBPILOT_QQ_MAIL_USERNAME", "").strip()
    auth_code = os.environ.get("JOBPILOT_QQ_MAIL_AUTH_CODE", "").strip()
    if not username or not auth_code:
        raise RuntimeError("QQ Mail credentials are not configured")
    mailbox = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT, timeout=20)
    try:
        status, _ = mailbox.login(username, auth_code)
        if status != "OK":
            raise RuntimeError("QQ Mail IMAPS authentication failed")
        status, _ = mailbox.select("INBOX", readonly=True)
        if status != "OK":
            raise RuntimeError("QQ Mail INBOX could not be opened read-only")
        return mailbox
    except Exception:
        try:
            mailbox.logout()
        except Exception:
            pass
        raise


def _validate_list_args(limit: int, since_days: int, subject: str | None, sender: str | None) -> None:
    if isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if isinstance(since_days, bool) or not 1 <= since_days <= 365:
        raise ValueError("since_days must be between 1 and 365")
    for value in (subject, sender):
        if value is not None and (not isinstance(value, str) or len(value) > 200):
            raise ValueError("mail filters must be strings of at most 200 characters")


def _validate_message_args(uid: str, include_body: bool, max_chars: int) -> None:
    if not isinstance(uid, str) or not _UID_PATTERN.fullmatch(uid):
        raise ValueError("uid must be a numeric IMAP UID")
    if not isinstance(include_body, bool):
        raise ValueError("include_body must be boolean")
    if isinstance(max_chars, bool) or not 1000 <= max_chars <= 50000:
        raise ValueError("max_chars must be between 1000 and 50000")


def _header_payload(uid: str, message: Message) -> dict[str, Any]:
    return {
        "uid": uid,
        "message_id": str(message.get("Message-ID") or "").strip(),
        "from": _decode_header(str(message.get("From") or "")),
        "to": _decode_header(str(message.get("To") or "")),
        "subject": _decode_header(str(message.get("Subject") or "")),
        "date": str(message.get("Date") or "").strip(),
    }


def _message_body(message: Message, *, max_chars: int) -> str:
    parts: list[str] = []
    if message.is_multipart():
        candidates = message.walk()
    else:
        candidates = (message,)
    for part in candidates:
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        if part.get_content_type() not in {"text/plain", "text/html"}:
            continue
        try:
            content = part.get_content()
        except Exception:
            raw = part.get_payload(decode=True) or b""
            content = raw.decode(part.get_content_charset() or "utf-8", "replace")
        text = str(content).strip()
        if part.get_content_type() == "text/html":
            text = re.sub(r"<[^>]+>", " ", text)
        if text:
            parts.append(text)
    return "\n\n".join(parts)[:max_chars]


def _decode_header(value: str) -> str:
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def _imap_text(value: str) -> str:
    return value.replace('\\', " ").replace('"', " ").replace("\r", " ").replace("\n", " ").strip()


def _first_bytes(data: Any) -> bytes | None:
    if not isinstance(data, (list, tuple)):
        return None
    for item in data:
        if isinstance(item, tuple) and len(item) > 1 and isinstance(item[1], bytes):
            return item[1]
    return None


def _flags_from_fetch(data: Any) -> list[str]:
    if not isinstance(data, (list, tuple)):
        return []
    for item in data:
        if not isinstance(item, tuple) or not item or not isinstance(item[0], bytes):
            continue
        match = re.search(rb"FLAGS \(([^)]*)\)", item[0])
        if match:
            return match.group(1).decode("ascii", "ignore").split()
    return []


if __name__ == "__main__":
    server.run("stdio")
