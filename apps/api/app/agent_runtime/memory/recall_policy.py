from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from sqlalchemy.orm import Session

from app.domains.agent_memory.models import AgentMemory, AgentMemoryStatus
from app.domains.agent_memory.repository import AgentMemoryRepository


@dataclass(frozen=True)
class RecalledMemoryItem:
    memory_id: str
    title: str
    content: str
    scope: str
    memory_type: str
    importance: int
    score: int


@dataclass(frozen=True)
class MemoryRecallResult:
    query: str
    items: list[RecalledMemoryItem]
    rendered_context: str
    truncated: bool = False


def recall_relevant_memories(
    session: Session,
    *,
    query: str,
    limit: int = 3,
    max_chars: int = 2000,
) -> MemoryRecallResult:
    """Select active long-term memories relevant to the current user turn.

    This is intentionally deterministic: the context builder can decide what to
    load without asking the model to inspect the whole memory store.
    """

    repository = AgentMemoryRepository(session)
    memories = repository.list_memories(status=AgentMemoryStatus.ACTIVE, limit=1000)
    return recall_relevant_memory_records(memories, query=query, limit=limit, max_chars=max_chars)


def recall_relevant_memory_records(
    memories: Sequence[AgentMemory],
    *,
    query: str,
    limit: int = 3,
    max_chars: int = 2000,
) -> MemoryRecallResult:
    normalized_query = _normalize(query)
    if not normalized_query or limit <= 0 or max_chars <= 0:
        return MemoryRecallResult(query=query, items=[], rendered_context="", truncated=False)

    terms = _query_terms(normalized_query)
    scored: list[tuple[int, AgentMemory]] = []
    for memory in memories:
        if _status_value(memory.status) != AgentMemoryStatus.ACTIVE.value:
            continue
        score = _memory_score(memory, terms=terms, normalized_query=normalized_query)
        if score > 0:
            scored.append((score, memory))

    scored.sort(key=lambda item: (item[0], item[1].importance, item[1].updated_at, item[1].id), reverse=True)
    selected = scored[:limit]
    items = [
        RecalledMemoryItem(
            memory_id=memory.id,
            title=memory.title,
            content=memory.content,
            scope=memory.scope,
            memory_type=memory.memory_type,
            importance=memory.importance,
            score=score,
        )
        for score, memory in selected
    ]
    rendered_context, char_truncated = _render_memory_context(items, max_chars=max_chars)
    return MemoryRecallResult(
        query=query,
        items=items,
        rendered_context=rendered_context,
        truncated=char_truncated or len(scored) > len(selected),
    )


def _memory_score(memory: AgentMemory, *, terms: set[str], normalized_query: str) -> int:
    searchable = _normalize("\n".join([memory.title, memory.content, memory.scope, memory.memory_type]))
    score = 0
    for term in terms:
        if not term:
            continue
        if term in searchable:
            score += 40
        if term in _normalize(memory.title):
            score += 20
        if term in _normalize(memory.scope):
            score += 10
    for alias in _scope_aliases(memory.scope):
        if alias in normalized_query:
            score += 35
    return score + max(0, min(int(memory.importance or 0), 100)) // 10 if score else 0


def _query_terms(normalized_query: str) -> set[str]:
    terms = {match.group(0).casefold() for match in re.finditer(r"[a-z0-9_+#.\-]{2,}", normalized_query)}
    for phrase in (
        "投递",
        "申请",
        "提交",
        "网申",
        "岗位",
        "职位",
        "招聘",
        "校招",
        "秋招",
        "春招",
        "简历",
        "确认",
        "面试",
        "搜索",
        "抓取",
        "公众号",
        "小红书",
        "文章",
    ):
        if phrase in normalized_query:
            terms.add(phrase)
    return terms


def _scope_aliases(scope: str) -> tuple[str, ...]:
    aliases = {
        "application_submission": ("投递", "申请", "提交", "网申"),
        "job_discovery": ("岗位", "职位", "招聘", "校招", "秋招", "春招"),
        "resume_tailoring": ("简历", "修改简历", "优化简历"),
        "content_fetcher": ("文章", "公众号", "小红书", "抓取"),
        "tool_recovery": ("工具", "失败", "恢复", "重试"),
    }
    return aliases.get(scope, ())


def _render_memory_context(items: Sequence[RecalledMemoryItem], *, max_chars: int) -> tuple[str, bool]:
    parts: list[str] = []
    truncated = False
    for item in items:
        entry = (
            f"- memory_id: {item.memory_id}\n"
            f"  title: {item.title}\n"
            f"  scope: {item.scope}\n"
            f"  type: {item.memory_type}\n"
            f"  content: {item.content}"
        )
        prefix = "\n\n" if parts else ""
        remaining = max_chars - len("".join(parts)) - len(prefix)
        if remaining <= 0:
            truncated = True
            break
        if len(entry) > remaining:
            entry = entry[: max(0, remaining - 3)].rstrip() + "..."
            truncated = True
        parts.append(prefix + entry)
    return "".join(parts), truncated


def _normalize(value: str | None) -> str:
    return " ".join(str(value or "").casefold().split())


def _status_value(status: AgentMemoryStatus | str) -> str:
    return str(getattr(status, "value", status))
