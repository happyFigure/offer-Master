from __future__ import annotations

from datetime import UTC, date, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.domains.applications.models import Application


def new_uuid() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class JobStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    ARCHIVED = "archived"


class JobSourceType(str, Enum):
    MANUAL_CLIP = "manual_clip"
    PUBLIC_ARTICLE = "public_article"
    XIAOHONGSHU_NOTE = "xiaohongshu_note"
    WECHAT_ARTICLE = "wechat_article"
    WECHAT_ACCOUNT = "wechat_account"
    UNIVERSITY_CAREER_SITE = "university_career_site"
    OFFICIAL_CAREER_SITE = "official_career_site"
    JOB_BOARD_VISIBLE_PAGE = "job_board_visible_page"
    OFFICIAL_API = "official_api"


class JobSourceFetchMode(str, Enum):
    MANUAL_CLIP = "manual_clip"
    PUBLIC_HTML = "public_html"
    MCP_VISIBLE_PAGE = "mcp_visible_page"
    OFFICIAL_API = "official_api"


class JobSourceTrustLevel(str, Enum):
    HIGH = "high"
    MEDIUM_HIGH = "medium_high"
    MEDIUM = "medium"
    LOW_MEDIUM = "low_medium"
    LOW = "low"


class SourceSyncRunStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"


class RawJobLeadStatus(str, Enum):
    CAPTURED = "captured"
    EXTRACTED = "extracted"
    FAILED = "failed"


class ArticleCandidateStatus(str, Enum):
    PENDING = "pending"
    PARSED = "parsed"
    SKIPPED = "skipped"
    NEEDS_VISIBLE_PAGE = "needs_visible_page"


class RecruitingSignalType(str, Enum):
    CAMPUS_RECRUITMENT_OPEN = "campus_recruitment_open"
    INTERNSHIP_OPEN = "internship_open"
    INFO_SUMMARY = "info_summary"


class RecruitingSignalStatus(str, Enum):
    NEEDS_JOB_ENRICHMENT = "needs_job_enrichment"
    JOB_FOUND = "job_found"
    NO_MATCHING_JOB = "no_matching_job"
    EXPIRED = "expired"


class JobLeadStatus(str, Enum):
    UNVERIFIED = "unverified"
    PENDING_REVIEW = "pending_review"
    VERIFIED = "verified"
    CONVERTED = "converted"
    EXPIRED = "expired"
    INVALID = "invalid"


class UrlImportRunStatus(str, Enum):
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED_RECOVERABLE = "failed_recoverable"
    FAILED_TERMINAL = "failed_terminal"
    DUPLICATE = "duplicate"


class DomainHealthState(str, Enum):
    UNKNOWN = "unknown"
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    website_url: Mapped[str | None] = mapped_column(String(1024))
    industry: Mapped[str | None] = mapped_column(String(255))
    city: Mapped[str | None] = mapped_column(String(128))
    country: Mapped[str | None] = mapped_column(String(128))
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    jobs: Mapped[list[Job]] = relationship(
        back_populates="company",
        cascade="all, delete-orphan",
    )


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("source", "source_job_id", name="uq_jobs_source_source_job_id"),
        Index("ix_jobs_company_id", "company_id"),
        Index("ix_jobs_source", "source"),
        Index("ix_jobs_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    company_id: Mapped[str] = mapped_column(ForeignKey("companies.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    city: Mapped[str | None] = mapped_column(String(128))
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    source_job_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(2048))
    job_type: Mapped[str | None] = mapped_column(String(128))
    salary_text: Mapped[str | None] = mapped_column(String(255))
    jd_text: Mapped[str | None] = mapped_column(Text)
    skills: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    date_posted: Mapped[date | None] = mapped_column(Date)
    match_score: Mapped[float | None] = mapped_column(Numeric(5, 2))
    status: Mapped[JobStatus] = mapped_column(String(32), nullable=False, default=JobStatus.OPEN)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    company: Mapped[Company] = relationship(back_populates="jobs", lazy="joined")
    applications: Mapped[list[Application]] = relationship(back_populates="job")


class JobSource(Base):
    __tablename__ = "job_sources"
    __table_args__ = (
        Index("ix_job_sources_enabled", "enabled"),
        Index("ix_job_sources_source_type", "source_type"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    source_type: Mapped[JobSourceType] = mapped_column(String(64), nullable=False)
    entry_url: Mapped[str | None] = mapped_column(String(2048))
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    sync_interval_hours: Mapped[int] = mapped_column(Integer, nullable=False, default=24)
    trust_level: Mapped[JobSourceTrustLevel] = mapped_column(
        String(32),
        nullable=False,
        default=JobSourceTrustLevel.MEDIUM,
    )
    fetch_mode: Mapped[JobSourceFetchMode] = mapped_column(
        String(64),
        nullable=False,
        default=JobSourceFetchMode.MANUAL_CLIP,
    )
    notes: Mapped[str | None] = mapped_column(Text)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    sync_runs: Mapped[list[SourceSyncRun]] = relationship(back_populates="source")
    raw_leads: Mapped[list[RawJobLead]] = relationship(back_populates="source")
    leads: Mapped[list[JobLead]] = relationship(back_populates="source")
    url_import_runs: Mapped[list[UrlImportRun]] = relationship(back_populates="source")
    article_candidates: Mapped[list[ArticleCandidate]] = relationship(back_populates="source")
    recruiting_signals: Mapped[list[RecruitingSignal]] = relationship(back_populates="source")


class SourceSyncRun(Base):
    __tablename__ = "source_sync_runs"
    __table_args__ = (
        Index("ix_source_sync_runs_source_id", "source_id"),
        Index("ix_source_sync_runs_status", "status"),
        Index("ix_source_sync_runs_started_at", "started_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    source_id: Mapped[str] = mapped_column(ForeignKey("job_sources.id"), nullable=False)
    status: Mapped[SourceSyncRunStatus] = mapped_column(
        String(32),
        nullable=False,
        default=SourceSyncRunStatus.RUNNING,
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    fetched_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    extracted_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    run_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    source: Mapped[JobSource] = relationship(back_populates="sync_runs")
    raw_leads: Mapped[list[RawJobLead]] = relationship(back_populates="sync_run")
    article_candidates: Mapped[list[ArticleCandidate]] = relationship(back_populates="sync_run")


class RawJobLead(Base):
    __tablename__ = "raw_job_leads"
    __table_args__ = (
        UniqueConstraint("source_id", "content_hash", name="uq_raw_job_leads_source_hash"),
        Index("ix_raw_job_leads_source_id", "source_id"),
        Index("ix_raw_job_leads_sync_run_id", "sync_run_id"),
        Index("ix_raw_job_leads_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    source_id: Mapped[str] = mapped_column(ForeignKey("job_sources.id"), nullable=False)
    sync_run_id: Mapped[str | None] = mapped_column(ForeignKey("source_sync_runs.id"))
    source_url: Mapped[str | None] = mapped_column(String(2048))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False, default="text/plain")
    raw_content: Mapped[str] = mapped_column(Text, nullable=False)
    extracted_text: Mapped[str | None] = mapped_column(Text)
    status: Mapped[RawJobLeadStatus] = mapped_column(
        String(32),
        nullable=False,
        default=RawJobLeadStatus.CAPTURED,
    )
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    source: Mapped[JobSource] = relationship(back_populates="raw_leads")
    sync_run: Mapped[SourceSyncRun | None] = relationship(back_populates="raw_leads")
    leads: Mapped[list[JobLead]] = relationship(back_populates="raw_lead")
    recruiting_signals: Mapped[list[RecruitingSignal]] = relationship(back_populates="raw_lead")


class ArticleCandidate(Base):
    __tablename__ = "article_candidates"
    __table_args__ = (
        UniqueConstraint("source_id", "url_hash", name="uq_article_candidates_source_url_hash"),
        Index("ix_article_candidates_source_id", "source_id"),
        Index("ix_article_candidates_sync_run_id", "sync_run_id"),
        Index("ix_article_candidates_status", "status"),
        Index("ix_article_candidates_source_account", "source_account"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    source_id: Mapped[str] = mapped_column(ForeignKey("job_sources.id"), nullable=False)
    sync_run_id: Mapped[str | None] = mapped_column(ForeignKey("source_sync_runs.id"))
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    url_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_account: Mapped[str | None] = mapped_column(String(255))
    published_at: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[ArticleCandidateStatus] = mapped_column(
        String(32),
        nullable=False,
        default=ArticleCandidateStatus.PENDING,
    )
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    source: Mapped[JobSource] = relationship(back_populates="article_candidates")
    sync_run: Mapped[SourceSyncRun | None] = relationship(back_populates="article_candidates")
    recruiting_signals: Mapped[list[RecruitingSignal]] = relationship(back_populates="article_candidate")


class RecruitingSignal(Base):
    __tablename__ = "recruiting_signals"
    __table_args__ = (
        UniqueConstraint("source_id", "signal_hash", name="uq_recruiting_signals_source_hash"),
        Index("ix_recruiting_signals_source_id", "source_id"),
        Index("ix_recruiting_signals_raw_lead_id", "raw_lead_id"),
        Index("ix_recruiting_signals_article_candidate_id", "article_candidate_id"),
        Index("ix_recruiting_signals_company_name", "company_name"),
        Index("ix_recruiting_signals_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    source_id: Mapped[str] = mapped_column(ForeignKey("job_sources.id"), nullable=False)
    raw_lead_id: Mapped[str | None] = mapped_column(ForeignKey("raw_job_leads.id"))
    article_candidate_id: Mapped[str | None] = mapped_column(ForeignKey("article_candidates.id"))
    signal_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    company_name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_company_name: Mapped[str] = mapped_column(String(255), nullable=False)
    signal_type: Mapped[RecruitingSignalType] = mapped_column(String(64), nullable=False)
    graduation_year: Mapped[str | None] = mapped_column(String(32))
    source_url: Mapped[str | None] = mapped_column(String(2048))
    original_source: Mapped[str | None] = mapped_column(String(255))
    confidence_score: Mapped[float | None] = mapped_column(Numeric(5, 2))
    trust_level: Mapped[JobSourceTrustLevel] = mapped_column(String(32), nullable=False)
    status: Mapped[RecruitingSignalStatus] = mapped_column(
        String(64),
        nullable=False,
        default=RecruitingSignalStatus.NEEDS_JOB_ENRICHMENT,
    )
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    source: Mapped[JobSource] = relationship(back_populates="recruiting_signals", lazy="joined")
    raw_lead: Mapped[RawJobLead | None] = relationship(back_populates="recruiting_signals")
    article_candidate: Mapped[ArticleCandidate | None] = relationship(back_populates="recruiting_signals")


class JobLead(Base):
    __tablename__ = "job_leads"
    __table_args__ = (
        UniqueConstraint("source_id", "lead_hash", name="uq_job_leads_source_hash"),
        Index("ix_job_leads_source_id", "source_id"),
        Index("ix_job_leads_raw_lead_id", "raw_lead_id"),
        Index("ix_job_leads_converted_job_id", "converted_job_id"),
        Index("ix_job_leads_verification_status", "verification_status"),
        Index("ix_job_leads_company_name", "company_name"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    source_id: Mapped[str] = mapped_column(ForeignKey("job_sources.id"), nullable=False)
    raw_lead_id: Mapped[str | None] = mapped_column(ForeignKey("raw_job_leads.id"))
    converted_job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"))
    lead_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    company_name: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    city: Mapped[str | None] = mapped_column(String(128))
    job_direction: Mapped[str | None] = mapped_column(String(128))
    graduation_year: Mapped[str | None] = mapped_column(String(32))
    source_url: Mapped[str | None] = mapped_column(String(2048))
    apply_url: Mapped[str | None] = mapped_column(String(2048))
    verified_url: Mapped[str | None] = mapped_column(String(2048))
    job_type: Mapped[str | None] = mapped_column(String(128))
    salary_text: Mapped[str | None] = mapped_column(String(255))
    jd_text: Mapped[str | None] = mapped_column(Text)
    skills: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    deadline: Mapped[date | None] = mapped_column(Date)
    confidence_score: Mapped[float | None] = mapped_column(Numeric(5, 2))
    trust_level: Mapped[JobSourceTrustLevel] = mapped_column(String(32), nullable=False)
    verification_status: Mapped[JobLeadStatus] = mapped_column(
        String(32),
        nullable=False,
        default=JobLeadStatus.UNVERIFIED,
    )
    verification_notes: Mapped[str | None] = mapped_column(Text)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime)
    converted_at: Mapped[datetime | None] = mapped_column(DateTime)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    source: Mapped[JobSource] = relationship(back_populates="leads", lazy="joined")
    raw_lead: Mapped[RawJobLead | None] = relationship(back_populates="leads")
    converted_job: Mapped[Job | None] = relationship(lazy="joined")


class UrlImportRun(Base):
    __tablename__ = "url_import_runs"
    __table_args__ = (
        Index("ix_url_import_runs_workflow_run_id", "workflow_run_id"),
        Index("ix_url_import_runs_source_id", "source_id"),
        Index("ix_url_import_runs_status", "status"),
        Index("ix_url_import_runs_current_stage", "current_stage"),
        Index("ix_url_import_runs_domain", "domain"),
        Index("ix_url_import_runs_normalized_url_hash", "normalized_url_hash"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    workflow_run_id: Mapped[str] = mapped_column(ForeignKey("workflow_runs.id"), nullable=False)
    source_id: Mapped[str | None] = mapped_column(ForeignKey("job_sources.id"))
    input_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    normalized_url: Mapped[str | None] = mapped_column(String(2048))
    normalized_url_hash: Mapped[str | None] = mapped_column(String(64))
    source_type: Mapped[JobSourceType | None] = mapped_column(String(64))
    domain: Mapped[str | None] = mapped_column(String(255))
    fetch_layer: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[UrlImportRunStatus] = mapped_column(
        String(32),
        nullable=False,
        default=UrlImportRunStatus.RUNNING,
    )
    current_stage: Mapped[str] = mapped_column(String(128), nullable=False, default="created")
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tool_call_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_call_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    next_action: Mapped[str | None] = mapped_column(String(128))
    raw_job_lead_id: Mapped[str | None] = mapped_column(ForeignKey("raw_job_leads.id"))
    extracted_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duplicate_of_run_id: Mapped[str | None] = mapped_column(ForeignKey("url_import_runs.id"))
    run_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)

    source: Mapped[JobSource | None] = relationship(back_populates="url_import_runs")
    raw_job_lead: Mapped[RawJobLead | None] = relationship(lazy="joined")


class DomainHealth(Base):
    __tablename__ = "domain_health_states"
    __table_args__ = (
        UniqueConstraint("domain", "tool_name", name="uq_domain_health_domain_tool"),
        Index("ix_domain_health_states_domain", "domain"),
        Index("ix_domain_health_states_state", "state"),
        Index("ix_domain_health_states_cooldown_until", "cooldown_until"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    domain: Mapped[str] = mapped_column(String(255), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    state: Mapped[DomainHealthState] = mapped_column(
        String(32),
        nullable=False,
        default=DomainHealthState.UNKNOWN,
    )
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    success_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error_code: Mapped[str | None] = mapped_column(String(64))
    last_error_message: Mapped[str | None] = mapped_column(Text)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime)
    cooldown_until: Mapped[datetime | None] = mapped_column(DateTime)
    half_open_probe_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )


# Register reverse-side ORM models when callers import only jobs.models.
from app.domains.applications import models as application_models  # noqa: E402,F401
