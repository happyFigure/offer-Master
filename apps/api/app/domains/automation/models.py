from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Index, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.domains.applications.models import Application
from app.domains.jobs.models import Job


def new_uuid() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class WorkflowRunStatus(str, Enum):
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    FAILED = "failed"
    FAILED_RECOVERABLE = "failed_recoverable"
    COMPLETED = "completed"
    CANCELED = "canceled"


class ToolCallStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


class ApprovalRequestStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELED = "canceled"


class AutomationRunStatus(str, Enum):
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"
    __table_args__ = (
        Index("ix_workflow_runs_workflow_type", "workflow_type"),
        Index("ix_workflow_runs_status", "status"),
        Index("ix_workflow_runs_related_job_id", "related_job_id"),
        Index("ix_workflow_runs_related_application_id", "related_application_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    workflow_type: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[WorkflowRunStatus] = mapped_column(
        String(32),
        nullable=False,
        default=WorkflowRunStatus.RUNNING,
    )
    current_step: Mapped[str | None] = mapped_column(String(128))
    user_goal: Mapped[str | None] = mapped_column(Text)
    related_job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"))
    related_application_id: Mapped[str | None] = mapped_column(ForeignKey("applications.id"))
    approval_request_id: Mapped[str | None] = mapped_column(String(36))
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)

    related_job: Mapped[Job | None] = relationship(lazy="joined")
    related_application: Mapped[Application | None] = relationship(lazy="joined")
    checkpoints: Mapped[list[WorkflowCheckpoint]] = relationship(
        back_populates="workflow_run",
        cascade="all, delete-orphan",
        order_by="WorkflowCheckpoint.created_at",
    )
    tool_call_logs: Mapped[list[ToolCallLog]] = relationship(
        back_populates="workflow_run",
        cascade="all, delete-orphan",
        order_by="ToolCallLog.created_at",
    )
    approval_request: Mapped[ApprovalRequest | None] = relationship(
        back_populates="workflow_run",
        cascade="all, delete-orphan",
        uselist=False,
    )
    automation_runs: Mapped[list[AutomationRun]] = relationship(
        back_populates="workflow_run",
        cascade="all, delete-orphan",
        order_by="AutomationRun.started_at",
    )


class WorkflowCheckpoint(Base):
    __tablename__ = "workflow_checkpoints"
    __table_args__ = (
        Index("ix_workflow_checkpoints_workflow_run_id", "workflow_run_id"),
        Index("ix_workflow_checkpoints_checkpoint_key", "checkpoint_key"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    workflow_run_id: Mapped[str] = mapped_column(ForeignKey("workflow_runs.id"), nullable=False)
    checkpoint_key: Mapped[str] = mapped_column(String(255), nullable=False)
    state: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)

    workflow_run: Mapped[WorkflowRun] = relationship(back_populates="checkpoints")


class ToolCallLog(Base):
    __tablename__ = "tool_call_logs"
    __table_args__ = (
        Index("ix_tool_call_logs_workflow_run_id", "workflow_run_id"),
        Index("ix_tool_call_logs_tool_name", "tool_name"),
        Index("ix_tool_call_logs_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    workflow_run_id: Mapped[str] = mapped_column(ForeignKey("workflow_runs.id"), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_group: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[ToolCallStatus] = mapped_column(String(32), nullable=False)
    input_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    output_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None]
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)

    workflow_run: Mapped[WorkflowRun] = relationship(back_populates="tool_call_logs")


class ApprovalRequest(Base):
    __tablename__ = "approval_requests"
    __table_args__ = (
        Index("ix_approval_requests_workflow_run_id", "workflow_run_id"),
        Index("ix_approval_requests_application_id", "application_id"),
        Index("ix_approval_requests_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    workflow_run_id: Mapped[str] = mapped_column(ForeignKey("workflow_runs.id"), nullable=False)
    application_id: Mapped[str | None] = mapped_column(ForeignKey("applications.id"))
    action_type: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[ApprovalRequestStatus] = mapped_column(
        String(32),
        nullable=False,
        default=ApprovalRequestStatus.PENDING,
    )
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    decision: Mapped[str | None] = mapped_column(String(64))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime)

    workflow_run: Mapped[WorkflowRun] = relationship(back_populates="approval_request")
    application: Mapped[Application | None] = relationship()


class AutomationRun(Base):
    __tablename__ = "automation_runs"
    __table_args__ = (
        Index("ix_automation_runs_workflow_run_id", "workflow_run_id"),
        Index("ix_automation_runs_application_id", "application_id"),
        Index("ix_automation_runs_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    workflow_run_id: Mapped[str] = mapped_column(ForeignKey("workflow_runs.id"), nullable=False)
    application_id: Mapped[str | None] = mapped_column(ForeignKey("applications.id"))
    browser_session_id: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[AutomationRunStatus] = mapped_column(
        String(32),
        nullable=False,
        default=AutomationRunStatus.RUNNING,
    )
    target_url: Mapped[str | None] = mapped_column(String(2048))
    last_screenshot_path: Mapped[str | None] = mapped_column(String(1024))
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)

    workflow_run: Mapped[WorkflowRun] = relationship(back_populates="automation_runs")
    application: Mapped[Application | None] = relationship()
