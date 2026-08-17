from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domains.jobs.models import JobStatus
from app.domains.jobs.models import (
    ArticleCandidateStatus,
    DomainHealthState,
    JobLeadStatus,
    JobSourceFetchMode,
    JobSourceTrustLevel,
    JobSourceType,
    RawJobLeadStatus,
    RecruitingSignalStatus,
    RecruitingSignalType,
    SourceSyncRunStatus,
    UrlImportRunStatus,
)


class ToolErrorCode(str, Enum):
    TOOL_NOT_ALLOWED = "TOOL_NOT_ALLOWED"
    TOOL_CIRCUIT_OPEN = "TOOL_CIRCUIT_OPEN"
    TOOL_BUDGET_EXCEEDED = "TOOL_BUDGET_EXCEEDED"
    LLM_BUDGET_EXCEEDED = "LLM_BUDGET_EXCEEDED"
    TIME_BUDGET_EXCEEDED = "TIME_BUDGET_EXCEEDED"
    FETCH_ATTEMPTS_EXCEEDED = "FETCH_ATTEMPTS_EXCEEDED"
    URL_NOT_ALLOWED = "URL_NOT_ALLOWED"
    SOURCE_TYPE_NOT_ALLOWED = "SOURCE_TYPE_NOT_ALLOWED"
    FETCH_TIMEOUT = "FETCH_TIMEOUT"
    FETCH_BLOCKED = "FETCH_BLOCKED"
    FETCH_FAILED = "FETCH_FAILED"
    CONTENT_TOO_SHORT = "CONTENT_TOO_SHORT"
    REQUIRES_JAVASCRIPT = "REQUIRES_JAVASCRIPT"
    CONTENT_EXTRACTION_FAILED = "CONTENT_EXTRACTION_FAILED"
    REQUIRES_MCP_VISIBLE_PAGE = "REQUIRES_MCP_VISIBLE_PAGE"
    LLM_EXTRACTION_FAILED = "LLM_EXTRACTION_FAILED"
    MCP_USER_CONFIRMATION_REQUIRED = "MCP_USER_CONFIRMATION_REQUIRED"
    DUPLICATE_URL = "DUPLICATE_URL"
    VALIDATION_FAILED = "VALIDATION_FAILED"


class ToolSuggestedNextAction(str, Enum):
    CONTINUE_WORKFLOW = "continue_workflow"
    RETRY_SAME_STAGE = "retry_same_stage"
    RETRY_WITH_NEXT_FETCHER = "retry_with_next_fetcher"
    WAIT_FOR_COOLDOWN = "wait_for_cooldown"
    REQUEST_USER_VISIBLE_PAGE = "request_user_visible_page"
    REQUEST_MANUAL_PASTE = "request_manual_paste"
    SKIP_DUPLICATE = "skip_duplicate"
    STOP_TERMINAL_FAILURE = "stop_terminal_failure"
    ENRICH_RECRUITING_SIGNAL = "enrich_recruiting_signal"


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


class JobSourceUpdate(BaseModel):
    name: str | None = None
    source_type: JobSourceType | None = None
    entry_url: str | None = None
    enabled: bool | None = None
    sync_interval_hours: int | None = Field(default=None, ge=1, le=24 * 30)
    trust_level: JobSourceTrustLevel | None = None
    fetch_mode: JobSourceFetchMode | None = None
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


class ArticleCandidateCreate(BaseModel):
    source_id: str
    sync_run_id: str | None = None
    title: str
    url: str
    source_account: str | None = None
    published_at: datetime | None = None
    status: ArticleCandidateStatus = ArticleCandidateStatus.PENDING
    raw_payload: dict[str, Any] | None = None


class RecruitingSignalCreate(BaseModel):
    source_id: str
    raw_lead_id: str | None = None
    article_candidate_id: str | None = None
    company_name: str
    signal_type: RecruitingSignalType = RecruitingSignalType.CAMPUS_RECRUITMENT_OPEN
    graduation_year: str | None = None
    source_url: str | None = None
    original_source: str | None = None
    confidence_score: float | None = None
    trust_level: JobSourceTrustLevel | None = None
    status: RecruitingSignalStatus = RecruitingSignalStatus.NEEDS_JOB_ENRICHMENT
    raw_payload: dict[str, Any] | None = None


class JobLeadVerification(BaseModel):
    verification_status: JobLeadStatus
    verified_url: str | None = None
    verification_notes: str | None = None


class ImportUrlRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    source_id: str | None = None
    source_hint: JobSourceType | None = None
    trust_level: JobSourceTrustLevel | None = None
    force_refresh: bool = False

    @field_validator("url", mode="before")
    @classmethod
    def normalize_http_url(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("URL must be a string")
        normalized = value.strip()
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return normalized


class ImportUrlAcceptedResponse(BaseModel):
    run_id: str
    status: UrlImportRunStatus
    current_stage: str
    domain_health_state: DomainHealthState = DomainHealthState.UNKNOWN
    message: str


class VisiblePageContentRequest(BaseModel):
    visible_text: str = Field(min_length=1)
    title: str | None = None
    final_url: str | None = None


class UrlImportRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    workflow_run_id: str
    source_id: str | None = None
    input_url: str
    normalized_url: str | None = None
    normalized_url_hash: str | None = None
    source_type: JobSourceType | None = None
    domain: str | None = None
    fetch_layer: str | None = None
    status: UrlImportRunStatus
    current_stage: str
    attempt_count: int
    tool_call_count: int
    llm_call_count: int
    error_code: str | None = None
    error_message: str | None = None
    next_action: str | None = None
    raw_job_lead_id: str | None = None
    raw_content_preview: str | None = None
    raw_extraction_method: str | None = None
    raw_image_count: int | None = None
    raw_image_parse_deferred: bool | None = None
    extracted_count: int
    duplicate_of_run_id: str | None = None
    run_metadata: dict[str, Any] | None = None
    started_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None


class DomainHealthRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    domain: str
    tool_name: str
    state: DomainHealthState
    failure_count: int
    success_count: int
    last_error_code: str | None = None
    last_error_message: str | None = None
    opened_at: datetime | None = None
    cooldown_until: datetime | None = None
    half_open_probe_count: int
    created_at: datetime
    updated_at: datetime


class DomainHealthListResponse(BaseModel):
    items: list[DomainHealthRead]


class ToolResult(BaseModel):
    ok: bool
    stage: str
    tool_name: str
    error_code: ToolErrorCode | str | None = None
    error_message: str | None = None
    retryable: bool = False
    suggested_next_action: ToolSuggestedNextAction | str | None = None
    error_details: dict[str, Any] = Field(default_factory=dict)
    cost: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)


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


class JobSourceSyncRequest(BaseModel):
    limit: int = Field(default=20, ge=1, le=100)


class JobSourceSyncResponse(BaseModel):
    sync_run_id: str
    status: SourceSyncRunStatus
    fetched_count: int
    extracted_count: int
    failed_count: int
    error: str | None = None
    raw_leads: list[RawJobLeadRead]
    leads: list[JobLeadRead]
    article_candidates: list[ArticleCandidateRead] = Field(default_factory=list)
    recruiting_signals: list[RecruitingSignalRead] = Field(default_factory=list)


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


class ArticleCandidateRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source_id: str
    sync_run_id: str | None = None
    title: str
    url: str
    url_hash: str
    source_account: str | None = None
    published_at: datetime | None = None
    status: ArticleCandidateStatus
    raw_payload: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class ArticleCandidateListResponse(BaseModel):
    items: list[ArticleCandidateRead]


class RecruitingSignalRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source_id: str
    raw_lead_id: str | None = None
    article_candidate_id: str | None = None
    signal_hash: str
    company_name: str
    normalized_company_name: str
    signal_type: RecruitingSignalType
    graduation_year: str | None = None
    source_url: str | None = None
    original_source: str | None = None
    confidence_score: float | None = None
    trust_level: JobSourceTrustLevel
    status: RecruitingSignalStatus
    raw_payload: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class RecruitingSignalListResponse(BaseModel):
    items: list[RecruitingSignalRead]


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


class OfferIOCompanyRead(BaseModel):
    name: str
    company_nature: str | None = None
    industry: str | None = None
    locations: str | None = None
    job_count: int
    updated_at: str | None = None
    raw_payload: dict[str, Any] | None = None


class OfferIOCompanyOpeningRead(BaseModel):
    id: str
    company_name: str
    company_nature: str | None = None
    industry: str | None = None
    batch: str | None = None
    target: str | None = None
    location: str | None = None
    positions: str | None = None
    update_date: str | None = None
    deadline: str | None = None
    apply_link: str | None = None
    has_written_test: str | None = None
    raw_payload: dict[str, Any] | None = None


class OfferIOJobRead(BaseModel):
    id: str
    title: str
    company: str
    location: str | None = None
    category: str | None = None
    job_type: str | None = None
    publish_date: str | None = None
    salary: str | None = None
    deadline: str | None = None
    department: str | None = None
    apply_link: str | None = None
    source: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    raw_payload: dict[str, Any] | None = None


class OfferIOCompanyListResponse(BaseModel):
    items: list[OfferIOCompanyRead]
    page: int
    page_size: int
    total: int
    total_pages: int


class OfferIOCompanyOpeningListResponse(BaseModel):
    items: list[OfferIOCompanyOpeningRead]
    page: int
    page_size: int
    total: int
    total_pages: int


class OfferIOJobListResponse(BaseModel):
    items: list[OfferIOJobRead]
    page: int
    page_size: int
    total: int
    total_pages: int
