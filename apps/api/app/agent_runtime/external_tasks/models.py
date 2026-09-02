from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Index, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.agent_runtime.external_tasks.schemas import ExternalAgentTaskStatus, ExternalTaskType
from app.db.base import Base


def new_uuid() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class ExternalAgentTask(Base):
    __tablename__ = "external_agent_tasks"
    __table_args__ = (
        Index("ix_external_agent_tasks_task_type", "task_type"),
        Index("ix_external_agent_tasks_status", "status"),
        Index("ix_external_agent_tasks_trace_id", "trace_id"),
        Index("ix_external_agent_tasks_updated_at", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_type: Mapped[ExternalTaskType] = mapped_column(String(128), nullable=False)
    status: Mapped[ExternalAgentTaskStatus] = mapped_column(String(32), nullable=False)
    trace_id: Mapped[str | None] = mapped_column(String(128))
    context_pack_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    output_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    blocked_reason: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)

    events: Mapped[list[ExternalAgentTaskEvent]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="ExternalAgentTaskEvent.created_at",
    )
    artifacts: Mapped[list[ExternalAgentArtifact]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="ExternalAgentArtifact.created_at",
    )


class ExternalAgentTaskEvent(Base):
    __tablename__ = "external_agent_task_events"
    __table_args__ = (
        Index("ix_external_agent_task_events_task_id", "task_id"),
        Index("ix_external_agent_task_events_event_type", "event_type"),
        Index("ix_external_agent_task_events_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    task_id: Mapped[str] = mapped_column(ForeignKey("external_agent_tasks.id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)

    task: Mapped[ExternalAgentTask] = relationship(back_populates="events")


class ExternalAgentArtifact(Base):
    __tablename__ = "external_agent_artifacts"
    __table_args__ = (
        Index("ix_external_agent_artifacts_task_id", "task_id"),
        Index("ix_external_agent_artifacts_artifact_type", "artifact_type"),
        Index("ix_external_agent_artifacts_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    task_id: Mapped[str] = mapped_column(ForeignKey("external_agent_tasks.id"), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(128), nullable=False)
    path_or_uri: Mapped[str] = mapped_column(String(2048), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(128))
    artifact_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)

    task: Mapped[ExternalAgentTask] = relationship(back_populates="artifacts")
