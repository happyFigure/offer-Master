from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask, StandardAgentResult


SDK_AGENT_TASK_SCHEMA_VERSION = "offer_master.sdk_agent_task.v1"
SDK_AGENT_RESULT_SCHEMA_VERSION = "offer_master.sdk_agent_result.v1"

SdkAgentResultStatus = Literal["succeeded", "failed", "needs_approval", "partial", "canceled"]


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class SdkToolApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_type: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    tool_input: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(min_length=1)
    risk_level: str = "high"
    suggested_user_message: str | None = None
    approval_request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("approval_type", "tool_name", "reason", "risk_level", mode="after")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = str(value or "").strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @field_validator("suggested_user_message", "approval_request_id", mode="after")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

    def to_metadata_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class SdkToolTraceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_count: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    approval_request_ids: list[str] = Field(default_factory=list)
    operation_refs: list[str] = Field(default_factory=list)
    last_error_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("approval_request_ids", "operation_refs", mode="after")
    @classmethod
    def clean_string_list(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("last_error_type", mode="after")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

    def to_metadata_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class SdkAgentTaskEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SDK_AGENT_TASK_SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    subagent_name: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    input_payload: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    risk_policy: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    expected_output: list[str] = Field(default_factory=list)
    max_turns: int = Field(default=8, ge=1)
    context_refs: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def from_agent_task(
        cls,
        task: AgentTask,
        context: AgentRuntimeContext,
        *,
        subagent_name: str,
        allowed_tools: list[str] | tuple[str, ...] = (),
        risk_policy: dict[str, Any] | None = None,
        max_turns: int = 8,
        metadata: dict[str, Any] | None = None,
    ) -> "SdkAgentTaskEnvelope":
        source_type = str(context.permission_scope.get("source_type") or "").strip()
        context_refs = {
            "session_id": context.session_id,
            "workflow_run_id": context.run_id,
        }
        if source_type:
            context_refs["source_type"] = source_type
        return cls(
            task_id=context.task_id,
            trace_id=context.run_id,
            capability_id=task.capability_id,
            subagent_name=subagent_name,
            goal=task.goal,
            input_payload=dict(task.input_payload),
            allowed_tools=_clean_string_list(allowed_tools),
            risk_policy=dict(risk_policy or {}),
            constraints=_clean_string_list(task.constraints),
            expected_output=_clean_string_list(task.expected_output),
            max_turns=max_turns,
            context_refs=context_refs,
            metadata={**dict(context.metadata), **dict(metadata or {})},
        )

    @field_validator("schema_version", "task_id", "trace_id", "capability_id", "subagent_name", "goal", mode="after")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = str(value or "").strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @field_validator("allowed_tools", "constraints", "expected_output", mode="after")
    @classmethod
    def clean_model_string_list(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)

    def to_metadata_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class SdkAgentResultEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SDK_AGENT_RESULT_SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    subagent_name: str = Field(min_length=1)
    status: SdkAgentResultStatus
    summary: str = Field(min_length=1)
    observation: str | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    proposed_actions: list[dict[str, Any]] = Field(default_factory=list)
    operation_refs: list[str] = Field(default_factory=list)
    resource_effects: list[dict[str, Any]] = Field(default_factory=list)
    trace_summary: SdkToolTraceSummary = Field(default_factory=SdkToolTraceSummary)
    approval_request: SdkToolApprovalRequest | None = None
    raw_trace_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", "task_id", "capability_id", "subagent_name", "summary", mode="after")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = str(value or "").strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @field_validator("observation", "raw_trace_ref", mode="after")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

    @field_validator("operation_refs", mode="after")
    @classmethod
    def clean_operation_refs(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)

    @field_validator("resource_effects", mode="after")
    @classmethod
    def clean_resource_effects(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Child agents use this contract to tell the outer runtime which files or
        # artifacts now exist. Drop malformed entries instead of failing the whole run.
        return [dict(item) for item in value if isinstance(item, dict)]

    @model_validator(mode="after")
    def validate_approval_contract(self) -> "SdkAgentResultEnvelope":
        if self.status == "needs_approval" and self.approval_request is None:
            raise ValueError("needs_approval results require approval_request")
        return self

    def to_standard_agent_result(self) -> StandardAgentResult:
        return StandardAgentResult(
            status=self._standard_status(),
            summary=self.summary,
            observation=self.observation or self._default_observation(),
            evidence=list(self.evidence),
            next_actions=self._next_actions(),
            raw_result=self.to_metadata_dict(),
            requires_user_action=self.status == "needs_approval",
        )

    def to_metadata_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def _standard_status(self) -> str:
        if self.status == "needs_approval":
            return "blocked"
        return self.status

    def _next_actions(self) -> list[str]:
        if self.status != "needs_approval" or self.approval_request is None:
            return []
        return [f"Approve or reject {self.approval_request.tool_name}"]

    def _default_observation(self) -> str:
        details = [self.summary]
        if self.trace_summary.tool_call_count:
            details.append(f"tool_call_count={self.trace_summary.tool_call_count}")
        if self.trace_summary.retry_count:
            details.append(f"retry_count={self.trace_summary.retry_count}")
        confidence = self.diagnostics.get("confidence")
        if confidence is not None:
            details.append(f"confidence={confidence}")
        if self.operation_refs:
            details.append("operation_refs=" + ",".join(self.operation_refs))
        return " | ".join(details)


def _clean_string_list(values: list[str] | tuple[str, ...]) -> list[str]:
    return [str(value).strip() for value in values if str(value).strip()]
