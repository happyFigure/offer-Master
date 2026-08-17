from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

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
    created_at: datetime
