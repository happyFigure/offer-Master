from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agent_runtime.contracts.base import ExecutionResultBase, TaskEnvelopeBase
from app.agent_runtime.external_tasks.schemas import (
    ExternalTaskCandidateProfileRef,
    ExternalTaskJobContext,
    FindApplyEntryTaskEnvelope,
)
from app.agent_runtime.tool_registry import APPLICATION_FIND_APPLY_ENTRY_TOOL


class BrowserTaskType(str, Enum):
    PREPARE_APPLICATION = "browser.prepare_application"


class BrowserExecutionStatus(str, Enum):
    PREPARED = "prepared"
    WAITING_USER = "waiting_user"
    BLOCKED = "blocked"
    FAILED = "failed"
    SUBMITTED = "submitted"


class UserSelectedResumeFileRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(min_length=1)
    display_name: str | None = None
    mime_type: str | None = None


class BrowserTaskEnvelope(TaskEnvelopeBase):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "offer_master.browser_task.v1"
    capability: str = APPLICATION_FIND_APPLY_ENTRY_TOOL
    task_type: BrowserTaskType = BrowserTaskType.PREPARE_APPLICATION
    risk_level: str = "medium"
    objective: str = "Open the application page, prepare common fields, and stop before final submit."
    job: ExternalTaskJobContext
    candidate_profile_ref: ExternalTaskCandidateProfileRef
    start_url: str | None = None
    selected_resume_file_ref: UserSelectedResumeFileRef | None = None
    stop_before_submit: bool = True
    allowed_actions: list[str] = Field(
        default_factory=lambda: [
            "web_search",
            "open_apply_page",
            "click_apply_button",
            "read_visible_page",
            "fill_common_fields",
            "upload_user_selected_resume",
            "generate_question_answers",
            "screenshot",
        ]
    )
    forbidden_actions: list[str] = Field(
        default_factory=lambda: [
            "submit_application",
            "final_submit",
            "create_account",
            "send_email",
            "change_resume_source_file",
            "read_unselected_local_files",
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
    output_schema: str = "BrowserExecutionResultV1"

    @classmethod
    def from_find_apply_entry_task(cls, envelope: FindApplyEntryTaskEnvelope) -> BrowserTaskEnvelope:
        return cls(
            task_id=envelope.task_id,
            trace_id=envelope.trace_id,
            objective="Open the official application page and prepare the application. Stop before final submit.",
            job=envelope.job,
            candidate_profile_ref=envelope.candidate_profile_ref,
            start_url=envelope.job.apply_url_candidate or envelope.job.source_url,
            context_refs={
                "job_id": envelope.job.job_id,
                "profile_id": envelope.candidate_profile_ref.profile_id,
                "resume_version_id": envelope.candidate_profile_ref.resume_version_id,
            },
            metadata={"source_task_type": str(envelope.task_type.value), "stop_before_submit": True},
        )


class BrowserExecutionResult(ExecutionResultBase):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "offer_master.browser_execution_result.v1"
    status: BrowserExecutionStatus
    current_url: str | None = None
    apply_url: str | None = None
    filled_fields: dict[str, str] = Field(default_factory=dict)
    generated_answers: dict[str, str] = Field(default_factory=dict)
    executed_actions: list[str] = Field(default_factory=list)
    submitted: bool = False
    blocked_reason: str | None = None

    @model_validator(mode="after")
    def validate_browser_result_contract(self) -> BrowserExecutionResult:
        if self.status == BrowserExecutionStatus.PREPARED and self.submitted:
            raise ValueError("prepared browser results cannot mark submitted=true")
        if self.requires_user_action and not (self.next_action or self.blocked_reason):
            raise ValueError("requires_user_action results require next_action or blocked_reason")
        return self
