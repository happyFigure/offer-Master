from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domains.jobs.models import JobStatus


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
