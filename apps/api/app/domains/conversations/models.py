from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, foreign, mapped_column, relationship

from app.db.base import Base


def new_uuid() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class AgentSessionStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    CLOSED = "closed"


class AgentMessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class AgentMessageKind(str, Enum):
    SYSTEM_TEXT = "system_text"
    USER_TEXT = "user_text"
    ASSISTANT_TEXT = "assistant_text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SUMMARY_NOTICE = "summary_notice"
    SYNTHETIC_ERROR = "synthetic_error"


class AgentMessageVisibilityScope(str, Enum):
    USER_VISIBLE = "user_visible"
    RUNTIME_ONLY = "runtime_only"
    INTERNAL = "internal"


class AgentMessageProvenanceKind(str, Enum):
    USER_INPUT = "user_input"
    AGENT_GENERATED = "agent_generated"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM_GENERATED = "system_generated"


class AgentSession(Base):
    __tablename__ = "agent_sessions"
    __table_args__ = (
        Index("ix_agent_sessions_status", "status"),
        Index("ix_agent_sessions_primary_intent", "primary_intent"),
        Index("ix_agent_sessions_last_message_at", "last_message_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    title: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[AgentSessionStatus] = mapped_column(
        String(32),
        nullable=False,
        default=AgentSessionStatus.ACTIVE,
    )
    primary_intent: Mapped[str | None] = mapped_column(String(128))
    current_agent_run_id: Mapped[str | None] = mapped_column(String(64))
    last_context_summary_id: Mapped[str | None] = mapped_column(String(36))
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    messages: Mapped[list[AgentMessage]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="AgentMessage.created_at",
    )
    context_summaries: Mapped[list[AgentContextSummary]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        foreign_keys="AgentContextSummary.session_id",
        order_by="AgentContextSummary.created_at",
    )
    last_context_summary: Mapped[AgentContextSummary | None] = relationship(
        "AgentContextSummary",
        primaryjoin=lambda: foreign(AgentSession.last_context_summary_id) == AgentContextSummary.id,
        post_update=True,
        uselist=False,
    )


class AgentMessage(Base):
    __tablename__ = "agent_messages"
    __table_args__ = (
        Index("ix_agent_messages_session_id", "session_id"),
        Index("ix_agent_messages_role", "role"),
        Index("ix_agent_messages_message_kind", "message_kind"),
        Index("ix_agent_messages_agent_run_id", "agent_run_id"),
        Index("ix_agent_messages_workflow_run_id", "workflow_run_id"),
        Index("ix_agent_messages_tool_call_log_id", "tool_call_log_id"),
        Index("ix_agent_messages_compacted_by_summary_id", "compacted_by_summary_id"),
        Index("ix_agent_messages_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("agent_sessions.id"), nullable=False)
    role: Mapped[AgentMessageRole] = mapped_column(String(32), nullable=False)
    message_kind: Mapped[AgentMessageKind] = mapped_column(String(64), nullable=False)
    agent_id: Mapped[str | None] = mapped_column(String(128))
    recipient_agent_id: Mapped[str | None] = mapped_column(String(128))
    visibility_scope: Mapped[AgentMessageVisibilityScope] = mapped_column(
        String(32),
        nullable=False,
        default=AgentMessageVisibilityScope.USER_VISIBLE,
    )
    content_text: Mapped[str | None] = mapped_column(Text)
    content_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    visible_content_text: Mapped[str | None] = mapped_column(Text)
    runtime_content_text: Mapped[str | None] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False, default="text/plain")
    provenance_kind: Mapped[AgentMessageProvenanceKind | None] = mapped_column(String(64))
    agent_run_id: Mapped[str | None] = mapped_column(String(64))
    workflow_run_id: Mapped[str | None] = mapped_column(ForeignKey("workflow_runs.id"))
    tool_call_log_id: Mapped[str | None] = mapped_column(ForeignKey("tool_call_logs.id"))
    parent_message_id: Mapped[str | None] = mapped_column(ForeignKey("agent_messages.id"))
    token_estimate: Mapped[int | None] = mapped_column(Integer)
    exclude_from_context: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    compacted_by_summary_id: Mapped[str | None] = mapped_column(
        ForeignKey("agent_context_summaries.id"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    session: Mapped[AgentSession] = relationship(back_populates="messages")
    parent_message: Mapped[AgentMessage | None] = relationship(remote_side="AgentMessage.id")
    compacted_by_summary: Mapped[AgentContextSummary | None] = relationship(
        "AgentContextSummary",
        foreign_keys=[compacted_by_summary_id],
        post_update=True,
    )


class AgentContextSummary(Base):
    __tablename__ = "agent_context_summaries"
    __table_args__ = (
        Index("ix_agent_context_summaries_session_id", "session_id"),
        Index("ix_agent_context_summaries_previous_summary_id", "previous_summary_id"),
        Index("ix_agent_context_summaries_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("agent_sessions.id"), nullable=False)
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    summary_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    covered_message_start_id: Mapped[str | None] = mapped_column(String(36))
    covered_message_end_id: Mapped[str | None] = mapped_column(String(36))
    first_kept_message_id: Mapped[str | None] = mapped_column(String(36))
    previous_summary_id: Mapped[str | None] = mapped_column(
        ForeignKey("agent_context_summaries.id"),
    )
    token_estimate: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    created_by: Mapped[str | None] = mapped_column(String(128))
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    session: Mapped[AgentSession] = relationship(
        back_populates="context_summaries",
        foreign_keys=[session_id],
    )
    covered_message_start: Mapped[AgentMessage | None] = relationship(
        "AgentMessage",
        primaryjoin=lambda: foreign(AgentContextSummary.covered_message_start_id) == AgentMessage.id,
        uselist=False,
    )
    covered_message_end: Mapped[AgentMessage | None] = relationship(
        "AgentMessage",
        primaryjoin=lambda: foreign(AgentContextSummary.covered_message_end_id) == AgentMessage.id,
        uselist=False,
    )
    first_kept_message: Mapped[AgentMessage | None] = relationship(
        "AgentMessage",
        primaryjoin=lambda: foreign(AgentContextSummary.first_kept_message_id) == AgentMessage.id,
        uselist=False,
    )
    previous_summary: Mapped[AgentContextSummary | None] = relationship(
        remote_side="AgentContextSummary.id",
    )
