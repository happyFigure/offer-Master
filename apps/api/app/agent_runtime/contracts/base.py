from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskEnvelopeBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    task_type: str = Field(min_length=1)
    risk_level: str = "low"
    created_at: datetime = Field(default_factory=utc_now)
    allowed_actions: list[str] = Field(default_factory=list)
    forbidden_actions: list[str] = Field(default_factory=list)
    human_approval_required: list[str] = Field(default_factory=list)
    context_refs: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionResultBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    summary: str | None = None
    observations: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    requires_user_action: bool = False
    next_action: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_user_action_reason(self) -> ExecutionResultBase:
        blocked_reason = getattr(self, "blocked_reason", None)
        if self.requires_user_action and not (self.next_action or blocked_reason):
            raise ValueError("requires_user_action results require next_action or blocked_reason")
        return self
