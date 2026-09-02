from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.agent_runtime.durable_state.schemas import (
    AgentArtifactSourceKind,
    AgentMemoryVisibilityScope,
    AgentStepStatus,
    AgentTaskStatus,
)
from app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class AgentTaskState(Base):
    __tablename__ = "agent_task_states"
    __table_args__ = (
        Index("ix_agent_task_states_root_workflow_run_id", "root_workflow_run_id"),
        Index("ix_agent_task_states_conversation_session_id", "conversation_session_id"),
        Index("ix_agent_task_states_task_type", "task_type"),
        Index("ix_agent_task_states_capability", "capability"),
        Index("ix_agent_task_states_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    root_workflow_run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    conversation_session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    task_type: Mapped[str] = mapped_column(String(128), nullable=False)
    capability: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[AgentTaskStatus] = mapped_column(String(32), nullable=False, default=AgentTaskStatus.CREATED)
    current_step_id: Mapped[str | None] = mapped_column(String(64))
    owner_executor: Mapped[str | None] = mapped_column(String(128))
    user_goal: Mapped[str | None] = mapped_column(String(2048))
    input_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    output_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now, onupdate=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)

    steps: Mapped[list[AgentStepState]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="AgentStepState.sequence_index",
    )
    memory_snapshots: Mapped[list[AgentMemorySnapshot]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="AgentMemorySnapshot.created_at",
    )
    artifacts: Mapped[list[AgentArtifactIndex]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="AgentArtifactIndex.created_at",
    )


class AgentStepState(Base):
    __tablename__ = "agent_step_states"
    __table_args__ = (
        Index("ix_agent_step_states_task_id", "task_id"),
        Index("ix_agent_step_states_status", "status"),
        Index("ix_agent_step_states_capability", "capability"),
        Index("ix_agent_step_states_external_task_id", "external_task_id"),
        Index("ix_agent_step_states_tool_call_log_id", "tool_call_log_id"),
        Index("ix_agent_step_states_approval_request_id", "approval_request_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("agent_task_states.id"), nullable=False)
    parent_step_id: Mapped[str | None] = mapped_column(String(64))
    sequence_index: Mapped[int] = mapped_column(nullable=False)
    step_type: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[AgentStepStatus] = mapped_column(String(32), nullable=False, default=AgentStepStatus.PENDING)
    executor_type: Mapped[str] = mapped_column(String(128), nullable=False)
    executor_name: Mapped[str] = mapped_column(String(128), nullable=False)
    capability: Mapped[str] = mapped_column(String(128), nullable=False)
    input_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    output_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    tool_call_log_id: Mapped[str | None] = mapped_column(String(64))
    external_task_id: Mapped[str | None] = mapped_column(String(64))
    approval_request_id: Mapped[str | None] = mapped_column(String(64))
    retry_count: Mapped[int] = mapped_column(nullable=False, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now, onupdate=utc_now)

    task: Mapped[AgentTaskState] = relationship(back_populates="steps")


class AgentMemorySnapshot(Base):
    __tablename__ = "agent_memory_snapshots"
    __table_args__ = (
        Index("ix_agent_memory_snapshots_task_id", "task_id"),
        Index("ix_agent_memory_snapshots_step_id", "step_id"),
        Index("ix_agent_memory_snapshots_memory_id", "memory_id"),
        Index("ix_agent_memory_snapshots_visibility_scope", "visibility_scope"),
        Index("ix_agent_memory_snapshots_passed_to_executor", "passed_to_executor"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("agent_task_states.id"), nullable=False)
    step_id: Mapped[str | None] = mapped_column(ForeignKey("agent_step_states.id"))
    memory_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_type: Mapped[str] = mapped_column(String(128), nullable=False)
    usage_reason: Mapped[str] = mapped_column(String(1024), nullable=False)
    visibility_scope: Mapped[AgentMemoryVisibilityScope] = mapped_column(
        String(64),
        nullable=False,
        default=AgentMemoryVisibilityScope.RUNTIME_ONLY,
    )
    passed_to_executor: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    memory_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)

    task: Mapped[AgentTaskState] = relationship(back_populates="memory_snapshots")
    step: Mapped[AgentStepState | None] = relationship()


class AgentArtifactIndex(Base):
    __tablename__ = "agent_artifact_index"
    __table_args__ = (
        Index("ix_agent_artifact_index_task_id", "task_id"),
        Index("ix_agent_artifact_index_step_id", "step_id"),
        Index("ix_agent_artifact_index_source_kind", "source_kind"),
        Index("ix_agent_artifact_index_artifact_type", "artifact_type"),
        Index("ix_agent_artifact_index_sequence_index", "sequence_index"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("agent_task_states.id"), nullable=False)
    step_id: Mapped[str | None] = mapped_column(ForeignKey("agent_step_states.id"))
    sequence_index: Mapped[int] = mapped_column(nullable=False, default=0)
    source_kind: Mapped[AgentArtifactSourceKind] = mapped_column(
        String(64),
        nullable=False,
        default=AgentArtifactSourceKind.RESULT_ENVELOPE,
    )
    artifact_type: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str | None] = mapped_column(String(512))
    uri: Mapped[str] = mapped_column(String(2048), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(128))
    artifact_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)

    task: Mapped[AgentTaskState] = relationship(back_populates="artifacts")
    step: Mapped[AgentStepState | None] = relationship()
