from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy import inspect, or_, select
from sqlalchemy.orm import Session

from app.domains.agent_memory.models import AgentMemory, AgentMemoryStatus, AgentSkill, AgentSkillStatus
from app.domains.conversations.models import AgentContextSummary, AgentMessage, AgentSession


MEMORY_TABLES = ["agent_memories", "agent_skills"]


@dataclass(frozen=True)
class SessionSearchItem:
    source_type: str
    session_id: str
    excerpt: str
    message_id: str | None = None
    summary_id: str | None = None
    title: str | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class SessionSearchResult:
    corpus: str
    query: str
    items: list[SessionSearchItem]


@dataclass(frozen=True)
class SessionHistoryMessage:
    id: str
    role: str
    message_kind: str
    content_text: str | None
    visible_content_text: str | None
    created_at: str


@dataclass(frozen=True)
class SessionHistoryWindow:
    session_id: str
    around_message_id: str | None
    messages: list[SessionHistoryMessage]
    truncated_before: bool
    truncated_after: bool


@dataclass(frozen=True)
class MemorySearchItem:
    memory_id: str
    source_type: str
    excerpt: str
    score: float | None = None


@dataclass(frozen=True)
class MemorySearchResult:
    corpus: str
    query: str
    searched_tables: list[str]
    available_tables: list[str]
    items: list[MemorySearchItem]
    reason: str | None = None


@dataclass(frozen=True)
class MemoryRead:
    memory_id: str
    found: bool
    source_type: str | None = None
    content: str | None = None
    excerpt: str | None = None
    reason: str | None = None
    metadata: dict = field(default_factory=dict)


def sessions_search(session: Session, *, query: str, limit: int = 10) -> SessionSearchResult:
    normalized_query = query.strip()
    if not normalized_query:
        return SessionSearchResult(corpus="sessions", query=query, items=[])

    pattern = f"%{normalized_query}%"
    items: list[SessionSearchItem] = []

    message_statement = (
        select(AgentMessage, AgentSession.title)
        .join(AgentSession, AgentSession.id == AgentMessage.session_id)
        .where(
            or_(
                AgentMessage.visible_content_text.ilike(pattern),
                AgentMessage.content_text.ilike(pattern),
            )
        )
        .order_by(AgentMessage.created_at.desc(), AgentMessage.id.desc())
        .limit(limit)
    )
    for message, title in session.execute(message_statement).all():
        items.append(
            SessionSearchItem(
                source_type="message",
                session_id=message.session_id,
                message_id=message.id,
                title=title,
                excerpt=_excerpt(message.visible_content_text or message.content_text or "", normalized_query),
                created_at=_iso(message.created_at),
            )
        )

    remaining = max(0, limit - len(items))
    if remaining:
        summary_statement = (
            select(AgentContextSummary, AgentSession.title)
            .join(AgentSession, AgentSession.id == AgentContextSummary.session_id)
            .where(AgentContextSummary.summary_text.ilike(pattern))
            .order_by(AgentContextSummary.created_at.desc(), AgentContextSummary.id.desc())
            .limit(remaining)
        )
        for summary, title in session.execute(summary_statement).all():
            items.append(
                SessionSearchItem(
                    source_type="summary",
                    session_id=summary.session_id,
                    summary_id=summary.id,
                    title=title,
                    excerpt=_excerpt(summary.summary_text, normalized_query),
                    created_at=_iso(summary.created_at),
                )
            )

    return SessionSearchResult(corpus="sessions", query=query, items=items[:limit])


def sessions_history(
    session: Session,
    *,
    session_key: str,
    around_message_id: str | None = None,
    window_before: int = 5,
    window_after: int = 5,
) -> SessionHistoryWindow:
    messages = list(
        session.scalars(
            select(AgentMessage)
            .where(AgentMessage.session_id == session_key)
            .order_by(AgentMessage.created_at.asc(), AgentMessage.id.asc())
        ).all()
    )
    if not messages:
        return SessionHistoryWindow(
            session_id=session_key,
            around_message_id=around_message_id,
            messages=[],
            truncated_before=False,
            truncated_after=False,
        )

    if around_message_id is None:
        center_index = len(messages) - 1
    else:
        center_index = next((index for index, message in enumerate(messages) if message.id == around_message_id), -1)
        if center_index < 0:
            raise ValueError(f"Agent message not found in session history: {around_message_id}")

    start_index = max(0, center_index - max(0, window_before))
    end_index = min(len(messages), center_index + max(0, window_after) + 1)
    return SessionHistoryWindow(
        session_id=session_key,
        around_message_id=around_message_id,
        messages=[_history_message(message) for message in messages[start_index:end_index]],
        truncated_before=start_index > 0,
        truncated_after=end_index < len(messages),
    )


def memory_search(
    session: Session,
    *,
    query: str,
    corpus: str | None = None,
    limit: int = 10,
) -> MemorySearchResult:
    available_tables = _available_memory_tables(session)
    if not available_tables:
        return MemorySearchResult(
            corpus=corpus or "memories",
            query=query,
            searched_tables=MEMORY_TABLES.copy(),
            available_tables=[],
            items=[],
            reason="memory tables are not initialized",
        )
    normalized_query = query.strip()
    if not normalized_query:
        return MemorySearchResult(
            corpus=corpus or "memories",
            query=query,
            searched_tables=MEMORY_TABLES.copy(),
            available_tables=available_tables,
            items=[],
            reason="empty memory query",
        )

    items: list[MemorySearchItem] = []
    pattern = f"%{normalized_query}%"

    if "agent_memories" in available_tables:
        memory_statement = (
            select(AgentMemory)
            .where(AgentMemory.status == AgentMemoryStatus.ACTIVE)
            .where(
                or_(
                    AgentMemory.title.ilike(pattern),
                    AgentMemory.content.ilike(pattern),
                    AgentMemory.scope.ilike(pattern),
                )
            )
            .order_by(AgentMemory.importance.desc(), AgentMemory.updated_at.desc())
            .limit(limit)
        )
        for memory in session.scalars(memory_statement).all():
            items.append(
                MemorySearchItem(
                    memory_id=memory.id,
                    source_type="agent_memory",
                    excerpt=_excerpt(f"{memory.title}\n{memory.content}", normalized_query),
                )
            )

    remaining = max(0, limit - len(items))
    if remaining and "agent_skills" in available_tables:
        skill_statement = (
            select(AgentSkill)
            .where(AgentSkill.status != AgentSkillStatus.ARCHIVED)
            .where(
                or_(
                    AgentSkill.name.ilike(pattern),
                    AgentSkill.title.ilike(pattern),
                    AgentSkill.description.ilike(pattern),
                    AgentSkill.category.ilike(pattern),
                )
            )
            .order_by(AgentSkill.pinned.desc(), AgentSkill.updated_at.desc())
            .limit(remaining)
        )
        for skill in session.scalars(skill_statement).all():
            items.append(
                MemorySearchItem(
                    memory_id=skill.id,
                    source_type="skill",
                    excerpt=_excerpt(f"{skill.title}\n{skill.description}\n{skill.category}", normalized_query),
                )
            )

    return MemorySearchResult(
        corpus=corpus or "memories",
        query=query,
        searched_tables=MEMORY_TABLES.copy(),
        available_tables=available_tables,
        items=items[:limit],
        reason=None if items else "memory search found no matches",
    )


def memory_get(session: Session, *, memory_id: str) -> MemoryRead:
    available_tables = _available_memory_tables(session)
    if not available_tables:
        return MemoryRead(
            memory_id=memory_id,
            found=False,
            reason="memory tables are not initialized",
            metadata={"searched_tables": MEMORY_TABLES.copy(), "available_tables": []},
        )
    if "agent_memories" in available_tables:
        memory = session.get(AgentMemory, memory_id)
        if memory is not None:
            return MemoryRead(
                memory_id=memory_id,
                found=True,
                source_type="agent_memory",
                content=memory.content,
                excerpt=_excerpt(memory.content, memory.title),
                metadata={
                    "title": memory.title,
                    "scope": memory.scope,
                    "status": str(getattr(memory.status, "value", memory.status)),
                    "searched_tables": MEMORY_TABLES.copy(),
                    "available_tables": available_tables,
                },
            )
    if "agent_skills" in available_tables:
        skill = session.get(AgentSkill, memory_id)
        if skill is not None:
            content = _read_skill_content(skill)
            return MemoryRead(
                memory_id=memory_id,
                found=True,
                source_type="skill",
                content=content,
                excerpt=_excerpt(f"{skill.title}\n{skill.description}", skill.title),
                metadata={
                    "name": skill.name,
                    "title": skill.title,
                    "category": skill.category,
                    "status": str(getattr(skill.status, "value", skill.status)),
                    "file_path": skill.file_path,
                    "searched_tables": MEMORY_TABLES.copy(),
                    "available_tables": available_tables,
                },
            )
    return MemoryRead(
        memory_id=memory_id,
        found=False,
        reason="memory not found",
        metadata={"searched_tables": MEMORY_TABLES.copy(), "available_tables": available_tables},
    )


def _available_memory_tables(session: Session) -> list[str]:
    inspector = inspect(session.connection())
    table_names = set(inspector.get_table_names())
    return [table for table in MEMORY_TABLES if table in table_names]


def _history_message(message: AgentMessage) -> SessionHistoryMessage:
    return SessionHistoryMessage(
        id=message.id,
        role=str(getattr(message.role, "value", message.role)),
        message_kind=str(getattr(message.message_kind, "value", message.message_kind)),
        content_text=message.content_text,
        visible_content_text=message.visible_content_text,
        created_at=_iso(message.created_at),
    )


def _excerpt(text: str, query: str, radius: int = 80) -> str:
    if not text:
        return ""
    lower_text = text.lower()
    lower_query = query.lower()
    index = lower_text.find(lower_query)
    if index < 0:
        return text[: radius * 2]
    start = max(0, index - radius)
    end = min(len(text), index + len(query) + radius)
    return text[start:end]


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _read_skill_content(skill: AgentSkill) -> str:
    if skill.file_path and Path(skill.file_path).is_file():
        return Path(skill.file_path).read_text(encoding="utf-8")
    return f"# {skill.title}\n\n{skill.description}"
