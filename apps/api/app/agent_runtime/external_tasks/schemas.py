from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class ExternalTaskType(str, Enum):
    FIND_APPLY_ENTRY = "find_apply_entry"


class ExternalAgentTaskStatus(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class ApplyEntryDiscoveryStatus(str, Enum):
    FOUND_OPENED = "found_opened"
    BLOCKED = "blocked"
    FAILED = "failed"


class ApplyEntryBlockedReason(str, Enum):
    LOGIN_REQUIRED = "login_required"
    CAPTCHA = "captcha"
    NO_APPLY_BUTTON = "no_apply_button"
    JOB_CLOSED = "job_closed"
    AMBIGUOUS_MULTIPLE_JOBS = "ambiguous_multiple_jobs"
    SENSITIVE_QUESTION = "sensitive_question"
    FINAL_SUBMIT = "final_submit"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


class ExternalTaskJobContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_url: str | None = None
    apply_url_candidate: str | None = None
    jd_summary: str | None = None


class ExternalTaskCandidateProfileRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(min_length=1)
    resume_version_id: str = Field(min_length=1)


class FindApplyEntryTaskEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "offer_master.find_apply_entry_task.v1"
    task_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    task_type: ExternalTaskType = ExternalTaskType.FIND_APPLY_ENTRY
    created_at: datetime = Field(default_factory=utc_now)
    objective: str = "Find the real application entry for this job and open it in Edge. Stop before final submit."
    job: ExternalTaskJobContext
    candidate_profile_ref: ExternalTaskCandidateProfileRef
    allowed_actions: list[str] = Field(
        default_factory=lambda: [
            "web_search",
            "open_browser",
            "click_apply_button",
            "read_visible_page",
            "screenshot",
        ]
    )
    forbidden_actions: list[str] = Field(
        default_factory=lambda: [
            "submit_application",
            "answer_sensitive_questions",
            "change_resume_source_file",
            "send_email",
            "create_account",
        ]
    )
    human_approval_required: list[str] = Field(
        default_factory=lambda: [
            "login_required",
            "captcha",
            "file_upload_uncertain",
            "sensitive_question",
            "final_submit",
        ]
    )
    output_schema: str = "ApplyEntryDiscoveryResultV1"
    evidence_required: list[str] = Field(
        default_factory=lambda: [
            "final_url",
            "button_text",
            "screenshot_path",
            "reasoning_summary",
        ]
    )


class ExternalAgentArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_type: str = Field(min_length=1)
    path_or_uri: str = Field(min_length=1)
    mime_type: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class ApplyEntryDiscoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "offer_master.apply_entry_result.v1"
    task_id: str = Field(min_length=1)
    status: ApplyEntryDiscoveryStatus
    confidence: float = Field(ge=0, le=1)
    company_name: str | None = None
    job_title: str | None = None
    source_url: str | None = None
    apply_url: str | None = None
    final_browser_url: str | None = None
    platform: str | None = None
    button_text: str | None = None
    requires_login: bool = False
    blocked_reason: ApplyEntryBlockedReason | None = None
    candidate_urls: list[str] = Field(default_factory=list)
    evidence_artifacts: list[ExternalAgentArtifactRef] = Field(default_factory=list)
    notes: str | None = None
    next_action: str | None = None

    @model_validator(mode="after")
    def validate_status_contract(self) -> ApplyEntryDiscoveryResult:
        if self.status == ApplyEntryDiscoveryStatus.FOUND_OPENED:
            if not (self.apply_url or self.final_browser_url):
                raise ValueError("found_opened results require apply_url or final_browser_url")
            if not self.evidence_artifacts:
                raise ValueError("found_opened results require evidence_artifacts")
        if self.status == ApplyEntryDiscoveryStatus.BLOCKED and self.blocked_reason is None:
            raise ValueError("blocked results require blocked_reason")
        return self


def __getattr__(name: str):
    if name in {"BrowserExecutionResult", "BrowserExecutionStatus", "BrowserTaskEnvelope", "BrowserTaskType", "UserSelectedResumeFileRef"}:
        from app.agent_runtime.contracts import tasks as browser_tasks

        return getattr(browser_tasks, name)
    raise AttributeError(name)
