from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PureWindowsPath
import re
from typing import Any


LOCAL_FILE_REFERENCE_RE = re.compile(
    r"[A-Za-z]:[\\/][^\r\n`\"<>]*?\.(?:tex|md|txt|pdf|docx|json|csv|yaml|yml)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ToolInputCompletionResult:
    tool_input: dict[str, Any]
    filled_fields: tuple[str, ...] = ()
    missing_required_fields: tuple[str, ...] = ()
    sources: dict[str, str] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "filled_fields": list(self.filled_fields),
            "missing_required_fields": list(self.missing_required_fields),
            "sources": dict(self.sources),
            "completed": bool(self.filled_fields),
        }


def complete_tool_input(
    *,
    tool_name: str,
    tool_input: dict[str, Any] | None,
    input_schema: dict[str, Any] | None,
    user_message: str,
    recent_user_context: str = "",
    context: dict[str, Any] | None = None,
) -> ToolInputCompletionResult:
    completed = dict(tool_input or {})
    schema = input_schema if isinstance(input_schema, dict) else {}
    required_fields = _required_fields(schema)
    filled_fields: list[str] = []
    sources: dict[str, str] = {}
    context = context or {}

    if _field_is_missing(completed, "path") and "path" in _schema_field_names(schema):
        path = _best_recent_path(user_message=user_message, recent_user_context=recent_user_context, context=context)
        if path:
            completed["path"] = path
            filled_fields.append("path")
            sources["path"] = "recent_user_context"

    if tool_name == "filesystem.replace_text":
        replacement = _replacement_pair(
            user_message=user_message,
            recent_user_context=recent_user_context,
            path=str(completed.get("path") or ""),
        )
        if _field_is_missing(completed, "old_text") and replacement.get("old_text"):
            completed["old_text"] = replacement["old_text"]
            filled_fields.append("old_text")
            sources["old_text"] = replacement.get("old_text_source", "recent_context")
        if _field_is_missing(completed, "new_text") and replacement.get("new_text"):
            completed["new_text"] = replacement["new_text"]
            filled_fields.append("new_text")
            sources["new_text"] = replacement.get("new_text_source", "user_message")

    if tool_name == "external.web_search" and "query" in _schema_field_names(schema):
        query_completion = _completed_web_search_query(
            current_query=str(completed.get("query") or ""),
            user_message=user_message,
            recent_user_context=recent_user_context,
        )
        if query_completion and query_completion != str(completed.get("query") or ""):
            completed["query"] = query_completion
            filled_fields.append("query")
            sources["query"] = "user_message_with_recent_context"

    missing_required = tuple(field for field in required_fields if _field_is_missing(completed, field))
    return ToolInputCompletionResult(
        tool_input=completed,
        filled_fields=tuple(_dedupe(filled_fields)),
        missing_required_fields=missing_required,
        sources=sources,
    )


def _required_fields(schema: dict[str, Any]) -> tuple[str, ...]:
    required = schema.get("required")
    if not isinstance(required, list):
        return ()
    return tuple(str(field) for field in required if str(field).strip())


def _schema_field_names(schema: dict[str, Any]) -> set[str]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return set(_required_fields(schema))
    return {str(name) for name in properties}


def _field_is_missing(payload: dict[str, Any], field_name: str) -> bool:
    if field_name not in payload:
        return True
    value = payload.get(field_name)
    return value is None or (isinstance(value, str) and not value.strip())


def _best_recent_path(*, user_message: str, recent_user_context: str, context: dict[str, Any]) -> str:
    direct_paths = _extract_local_file_references(user_message)
    if direct_paths:
        return direct_paths[-1]

    context_paths = context.get("recent_file_paths")
    if isinstance(context_paths, list):
        for path in reversed(context_paths):
            cleaned = str(path or "").strip()
            if cleaned:
                return cleaned

    recent_paths = _extract_local_file_references(recent_user_context)
    return recent_paths[-1] if recent_paths else ""


def _extract_local_file_references(text: str) -> list[str]:
    if not text:
        return []
    return _dedupe(match.group(0).strip() for match in LOCAL_FILE_REFERENCE_RE.finditer(text))


def _replacement_pair(*, user_message: str, recent_user_context: str, path: str) -> dict[str, str]:
    combined = "\n".join(part for part in (recent_user_context, user_message) if part)
    pair = _explicit_replacement_pair(combined)
    if not pair:
        new_text = _new_name_from_text(combined)
        pair = {"new_text": new_text, "new_text_source": "user_message"} if new_text else {}
    if not pair.get("old_text"):
        inferred_old = _old_name_from_path(path)
        if inferred_old:
            pair = {**pair, "old_text": inferred_old, "old_text_source": "file_name"}
    return pair


def _explicit_replacement_pair(text: str) -> dict[str, str]:
    patterns = (
        r"(?:把|将)\s*[「“\"']?(?P<old>[\w\u4e00-\u9fff·.&\-]{1,40})[」”\"']?\s*(?:替换|换|改)(?:成|为)\s*[「“\"']?(?P<new>[\w\u4e00-\u9fff·.&\-]{1,40})",
        r"[「“\"'](?P<old>[^「」“”\"']{1,40})[」”\"']\s*(?:替换|换|改)(?:成|为)\s*[「“\"'](?P<new>[^「」“”\"']{1,40})[」”\"']",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        for match in reversed(matches):
            old_text = _clean_replacement_token(match.group("old"))
            new_text = _clean_replacement_token(match.group("new"))
            if old_text and new_text and old_text not in {"简历", "名字", "姓名", "简历名字"}:
                return {
                    "old_text": old_text,
                    "new_text": new_text,
                    "old_text_source": "user_message",
                    "new_text_source": "user_message",
                }
    return {}


def _new_name_from_text(text: str) -> str:
    patterns = (
        r"(?:名字|姓名)[^\r\n。；;,，]{0,12}(?:修改|改|换)(?:成|为)\s*[「“\"']?(?P<new>[\w\u4e00-\u9fff·.&\-]{1,40})",
        r"(?:换成|换为|改成|改为)\s*[「“\"']?(?P<new>[\w\u4e00-\u9fff·.&\-]{1,40})",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        for match in reversed(matches):
            new_text = _clean_replacement_token(match.group("new"))
            if new_text and new_text not in {"我", "一下", "内容", "文件"}:
                return new_text
    return ""


def _old_name_from_path(path: str) -> str:
    if not path:
        return ""
    filename = PureWindowsPath(path).name or path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = filename.rsplit(".", 1)[0]
    candidate = re.split(r"[-_\s]", stem, maxsplit=1)[0].strip()
    if not candidate or len(candidate) > 12:
        return ""
    if any("\u4e00" <= char <= "\u9fff" for char in candidate):
        return candidate
    return ""


def _clean_replacement_token(value: str) -> str:
    return str(value or "").strip().strip("：:，,。！？?；;、 ）)】]」”\"'")


def _completed_web_search_query(*, current_query: str, user_message: str, recent_user_context: str) -> str:
    query = current_query.strip() or user_message.strip()
    if not query:
        return ""
    subject = _recent_reference_subject(recent_user_context)
    if not subject:
        return query
    if subject.casefold() in query.casefold():
        return query
    if not _contains_contextual_pronoun(query) and current_query.strip():
        return query
    return f"{subject} {_remove_contextual_pronouns(query)}".strip()


def _contains_contextual_pronoun(text: str) -> bool:
    return any(marker in text for marker in ("它", "他们", "它们", "这个", "这家", "该公司", "刚才", "上面"))


def _remove_contextual_pronouns(text: str) -> str:
    result = text
    for marker in ("该公司", "这个公司", "这家公司", "这个", "这家", "它们", "他们", "它", "刚才", "上面"):
        result = result.replace(marker, "")
    return re.sub(r"\s+", " ", result).strip()


def _recent_reference_subject(recent_user_context: str) -> str:
    if not recent_user_context:
        return ""
    company_patterns = (
        r"(?:了解|查询|查一下|看看|关于|搜索|搜一下)?\s*(?P<name>[A-Za-z][A-Za-z0-9 .&\-]{1,60}?)(?:\s+这个|\s+这家)?\s*(?:公司|企业)",
        r"(?:了解|查询|查一下|看看|关于|搜索|搜一下)?\s*(?P<name>[\u4e00-\u9fffA-Za-z0-9·.&\- ]{2,40}?)(?:这个|这家)?(?:公司|企业)",
    )
    for pattern in company_patterns:
        matches = list(re.finditer(pattern, recent_user_context, flags=re.IGNORECASE))
        for match in reversed(matches):
            candidate = _clean_subject(match.group("name"))
            if candidate:
                return candidate
    latin_matches = re.findall(r"\b[A-Z][A-Za-z0-9]+(?:[ .&\-]+[A-Z][A-Za-z0-9]+)*\.?\b", recent_user_context)
    return _clean_subject(latin_matches[-1]) if latin_matches else ""


def _clean_subject(value: str) -> str:
    cleaned = str(value or "").strip(" ：:，,。！？?；;、的")
    cleaned = re.sub(r"^(?:我想|我想了解|了解|查询|查一下|看看|关于|搜索|搜一下)", "", cleaned).strip()
    if cleaned in {"这个", "这家", "公司", "企业", "它", "他们", "它们"}:
        return ""
    return cleaned


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


__all__ = ["ToolInputCompletionResult", "complete_tool_input"]
