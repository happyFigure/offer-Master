from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Index, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.domains.jobs.models import Job


def new_uuid() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class ApplicationStatus(str, Enum):
    EVALUATING = "evaluating"
    PREPARING = "preparing"
    APPLIED = "applied"
    WRITTEN_TEST = "written_test"
    INTERVIEW_1 = "interview_1"
    INTERVIEW_2 = "interview_2"
    HR_INTERVIEW = "hr_interview"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class Application(Base):
    __tablename__ = "applications"
    __table_args__ = (
        Index("ix_applications_job_id", "job_id"),
        Index("ix_applications_status", "status"),
        Index("ix_applications_next_follow_up_at", "next_follow_up_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), nullable=False)
    status: Mapped[ApplicationStatus] = mapped_column(
        String(32),
        nullable=False,
        default=ApplicationStatus.EVALUATING,
    )
    priority: Mapped[str] = mapped_column(String(32), nullable=False, default="medium")
    channel: Mapped[str | None] = mapped_column(String(128))
    applied_at: Mapped[datetime | None] = mapped_column(DateTime)
    next_follow_up_at: Mapped[datetime | None] = mapped_column(DateTime)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    job: Mapped[Job] = relationship(back_populates="applications", lazy="joined")
    events: Mapped[list[ApplicationEvent]] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
        order_by="ApplicationEvent.created_at",
        lazy="selectin",
    )


class ApplicationEvent(Base):
    __tablename__ = "application_events"
    __table_args__ = (
        Index("ix_application_events_application_id", "application_id"),
        Index("ix_application_events_event_type", "event_type"),
        Index("ix_application_events_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    from_status: Mapped[ApplicationStatus | None] = mapped_column(String(32))
    to_status: Mapped[ApplicationStatus | None] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    body: Mapped[str | None] = mapped_column(Text)
    actor: Mapped[str] = mapped_column(String(64), nullable=False, default="system")
    source: Mapped[str] = mapped_column(String(64), nullable=False, default="domain")
    event_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)

    application: Mapped[Application] = relationship(back_populates="events")
