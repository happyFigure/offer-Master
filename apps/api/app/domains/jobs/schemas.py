from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domains.jobs.models import JobStatus
from app.domains.jobs.models import (
    JobLeadStatus,
    JobSourceFetchMode,
    JobSourceTrustLevel,
    JobSourceType,
    RawJobLeadStatus,
    SourceSyncRunStatus,
)


class CompanyCreate(BaseModel):
    name: str
    normalized_name: str | None = None
    website_url: str | None = None
    industry: str | None = None
    city: str | None = None
    country: str | None = None
    raw_payload: dict[str, Any] | None = None


class JobImportDraft(BaseModel):
    company_name: str
    company_website_url: str | None = None
    company_industry: str | None = None
    company_city: str | None = None
    company_country: str | None = None
    title: str
    city: str | None = None
    source: str
    source_job_id: str
    source_url: str | None = None
    job_type: str | None = None
    salary_text: str | None = None
    jd_text: str | None = None
    skills: list[str] = Field(default_factory=list)
    date_posted: date | None = None
    match_score: float | None = None
    status: JobStatus = JobStatus.OPEN
    raw_payload: dict[str, Any] | None = None


class JobSourceCreate(BaseModel):
    name: str
    source_type: JobSourceType
    entry_url: str | None = None
    enabled: bool = True
    sync_interval_hours: int = Field(default=24, ge=1, le=24 * 30)
    trust_level: JobSourceTrustLevel = JobSourceTrustLevel.MEDIUM
    fetch_mode: JobSourceFetchMode = JobSourceFetchMode.MANUAL_CLIP
    notes: str | None = None
    raw_payload: dict[str, Any] | None = None


class SourceSyncRunCreate(BaseModel):
    source_id: str
    status: SourceSyncRunStatus = SourceSyncRunStatus.RUNNING
    fetched_count: int = 0
    extracted_count: int = 0
    failed_count: int = 0
    error: str | None = None
    run_metadata: dict[str, Any] | None = None


class RawJobLeadCreate(BaseModel):
    source_id: str
    sync_run_id: str | None = None
    source_url: str | None = None
    raw_content: str
    content_type: str = "text/plain"
    extracted_text: str | None = None
    status: RawJobLeadStatus = RawJobLeadStatus.CAPTURED
    raw_payload: dict[str, Any] | None = None


class JobLeadCreate(BaseModel):
    source_id: str
    raw_lead_id: str | None = None
    company_name: str
    title: str
    city: str | None = None
    job_direction: str | None = None
    graduation_year: str | None = None
    source_url: str | None = None
    apply_url: str | None = None
    job_type: str | None = None
    salary_text: str | None = None
    jd_text: str | None = None
    skills: list[str] = Field(default_factory=list)
    deadline: date | None = None
    confidence_score: float | None = None
    trust_level: JobSourceTrustLevel | None = None
    verification_status: JobLeadStatus = JobLeadStatus.UNVERIFIED
    raw_payload: dict[str, Any] | None = None


class JobLeadVerification(BaseModel):
    verification_status: JobLeadStatus
    verified_url: str | None = None
    verification_notes: str | None = None


class JobSourceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    source_type: JobSourceType
    entry_url: str | None = None
    enabled: bool
    sync_interval_hours: int
    trust_level: JobSourceTrustLevel
    fetch_mode: JobSourceFetchMode
    notes: str | None = None
    last_synced_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class JobSourceListResponse(BaseModel):
    items: list[JobSourceRead]


class RawJobLeadRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source_id: str
    sync_run_id: str | None = None
    source_url: str | None = None
    content_hash: str
    content_type: str
    raw_content: str
    extracted_text: str | None = None
    status: RawJobLeadStatus
    created_at: datetime
    updated_at: datetime


class RawJobLeadCaptureResponse(BaseModel):
    raw_lead: RawJobLeadRead
    created: bool


class JobLeadRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source_id: str
    raw_lead_id: str | None = None
    converted_job_id: str | None = None
    company_name: str
    title: str
    city: str | None = None
    job_direction: str | None = None
    graduation_year: str | None = None
    source_url: str | None = None
    apply_url: str | None = None
    verified_url: str | None = None
    job_type: str | None = None
    salary_text: str | None = None
    jd_text: str | None = None
    skills: list[str]
    deadline: date | None = None
    confidence_score: float | None = None
    trust_level: JobSourceTrustLevel
    verification_status: JobLeadStatus
    verification_notes: str | None = None
    verified_at: datetime | None = None
    converted_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class JobLeadListResponse(BaseModel):
    items: list[JobLeadRead]


class JobLeadExtractionRequest(BaseModel):
    source_id: str
    raw_content: str
    source_url: str | None = None
    content_type: str = "text/plain"
    sync_run_id: str | None = None


class JobLeadExtractionResponse(BaseModel):
    raw_lead: RawJobLeadRead
    raw_created: bool
    extracted_count: int
    leads: list[JobLeadRead]


class CompanyRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    normalized_name: str
    website_url: str | None = None
    industry: str | None = None
    city: str | None = None
    country: str | None = None
    created_at: datetime
    updated_at: datetime


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    company_id: str
    title: str
    city: str | None = None
    source: str
    source_job_id: str
    source_url: str | None = None
    job_type: str | None = None
    salary_text: str | None = None
    jd_text: str | None = None
    skills: list[str]
    date_posted: date | None = None
    match_score: float | None = None
    status: JobStatus
    created_at: datetime
    updated_at: datetime


class JobSyncRequest(BaseModel):
    keyword: str | None = None
    city: str | None = None
    remote_type: str = "onsite"
    job_type: str = "campus"
    sources: list[str] = Field(default_factory=lambda: ["tencent_campus"])
    limit: int = Field(default=20, ge=1, le=100)


class CompanySummaryRead(BaseModel):
    id: str
    name: str


class JobSummaryRead(BaseModel):
    id: str
    title: str
    company: CompanySummaryRead
    city: str | None = None
    source: str
    source_job_id: str
    source_url: str | None = None
    job_type: str | None = None
    skills: list[str]
    status: JobStatus


class JobLeadConversionResponse(BaseModel):
    lead: JobLeadRead
    job: JobSummaryRead
    created: bool


class JobSyncResponse(BaseModel):
    requested_sources: list[str]
    imported: int
    duplicates: int
    failed: int
    jobs: list[JobSummaryRead]
