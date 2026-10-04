from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domains.applications.models import ApplicationStatus
from app.domains.jobs.schemas import JobImportDraft, JobSummaryRead


class ApplicationCreate(BaseModel):
    job_id: str
    status: ApplicationStatus = ApplicationStatus.PREPARING
    priority: str = "medium"
    channel: str | None = None
    applied_at: datetime | None = None
    next_follow_up_at: datetime | None = None
    notes: str | None = None


class ApplicationFromJobCreate(BaseModel):
    job: JobImportDraft
    status: ApplicationStatus = ApplicationStatus.EVALUATING
    priority: str = "medium"
    channel: str | None = None
    applied_at: datetime | None = None
    next_follow_up_at: datetime | None = None
    notes: str | None = None


class ApplicationUpdate(BaseModel):
    status: ApplicationStatus | None = None
    priority: str | None = None
    channel: str | None = None
    applied_at: datetime | None = None
    next_follow_up_at: datetime | None = None
    notes: str | None = None
    actor: str = "user"
    source: str = "manual"


class ApplicationEventCreate(BaseModel):
    application_id: str
    event_type: str
    from_status: ApplicationStatus | None = None
    to_status: ApplicationStatus | None = None
    title: str
    body: str | None = None
    actor: str = "system"
    source: str = "domain"
    event_metadata: dict[str, Any] | None = None


class ApplicationMailNotificationCreate(BaseModel):
    application_id: str
    event_type: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=255)
    body: str | None = None
    company_name: str | None = Field(default=None, max_length=255)
    job_title: str | None = Field(default=None, max_length=255)
    scheduled_at: datetime | None = None
    deadline_at: datetime | None = None
    timezone: str | None = Field(default=None, max_length=64)
    join_url: str | None = Field(default=None, max_length=2048)
    confidence: float = Field(ge=0, le=1)
    source_message_id: str = Field(min_length=1, max_length=512)
    source_uid: str | None = Field(default=None, max_length=128)
    evidence: list[str] = Field(default_factory=list, max_length=3)
    to_status: ApplicationStatus | None = None
    model_name: str | None = Field(default=None, max_length=128)
    parser_version: str | None = Field(default=None, max_length=64)


class ApplicationMailSyncCreate(BaseModel):
    """Structured mail evidence used to create or update a board application."""

    company_name: str = Field(min_length=1, max_length=255)
    job_title: str | None = Field(default=None, max_length=255)
    event_type: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=255)
    body: str | None = None
    source_sent_at: datetime | None = None
    scheduled_at: datetime | None = None
    deadline_at: datetime | None = None
    deadline_offset_hours: int | None = Field(default=None, ge=1, le=720)
    timezone: str | None = Field(default=None, max_length=64)
    join_url: str | None = Field(default=None, max_length=2048)
    confidence: float = Field(ge=0, le=1)
    source_message_id: str = Field(min_length=1, max_length=512)
    source_uid: str | None = Field(default=None, max_length=128)
    evidence: list[str] = Field(default_factory=list, max_length=3)
    to_status: ApplicationStatus | None = None
    model_name: str | None = Field(default=None, max_length=128)
    parser_version: str | None = Field(default=None, max_length=64)

    @field_validator("company_name", mode="before")
    @classmethod
    def require_company_name(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("company_name is required for mail synchronization")
        return value.strip()

    @model_validator(mode="after")
    def require_sent_time_for_relative_deadline(self) -> "ApplicationMailSyncCreate":
        if self.deadline_offset_hours is not None and self.source_sent_at is None:
            raise ValueError("source_sent_at is required when deadline_offset_hours is provided")
        return self


class ApplicationMailTimingRead(BaseModel):
    event_type: str
    title: str
    source_sent_at: datetime | None = None
    scheduled_at: datetime | None = None
    deadline_at: datetime | None = None
    deadline_offset_hours: int | None = None
    timezone: str | None = None
    timing_source: str = "none"
    timing_note: str | None = None


class ApplicationMailNotificationUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    body: str | None = None
    company_name: str | None = Field(default=None, max_length=255)
    job_title: str | None = Field(default=None, max_length=255)
    scheduled_at: datetime | None = None
    deadline_at: datetime | None = None
    timezone: str | None = Field(default=None, max_length=64)
    join_url: str | None = Field(default=None, max_length=2048)
    to_status: ApplicationStatus | None = None
    evidence: list[str] | None = Field(default=None, max_length=3)


class ApplicationMailNotificationConfirm(BaseModel):
    to_status: ApplicationStatus | None = None


class ApplicationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    job_id: str
    status: ApplicationStatus
    priority: str
    channel: str | None = None
    applied_at: datetime | None = None
    next_follow_up_at: datetime | None = None
    notes: str | None = None
    created_at: datetime
    updated_at: datetime


class ApplicationBoardItem(ApplicationRead):
    job: JobSummaryRead
    mail_timing: ApplicationMailTimingRead | None = None


class ApplicationListResponse(BaseModel):
    items: list[ApplicationBoardItem]


class ApplicationEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    application_id: str
    event_type: str
    from_status: ApplicationStatus | None = None
    to_status: ApplicationStatus | None = None
    title: str
    body: str | None = None
    actor: str
    source: str
    event_metadata: dict[str, Any] | None = None
    scheduled_at: datetime | None = None
    deadline_at: datetime | None = None
    timezone: str | None = None
    join_url: str | None = None
    source_message_id: str | None = None
    source_uid: str | None = None
    review_status: str | None = None
    reviewed_at: datetime | None = None
    created_at: datetime


class ApplicationNotificationRead(ApplicationEventRead):
    application: ApplicationBoardItem


class ApplicationNotificationListResponse(BaseModel):
    items: list[ApplicationNotificationRead]
