from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.domains.automation.models import (
    ApprovalRequestStatus,
    ToolCallStatus,
    WorkflowRunStatus,
)


class WorkflowRunCreate(BaseModel):
    workflow_type: str
    current_step: str | None = None
    user_goal: str | None = None
    related_job_id: str | None = None
    related_application_id: str | None = None


class WorkflowCheckpointCreate(BaseModel):
    workflow_run_id: str
    checkpoint_key: str
    state: dict[str, Any]


class ToolCallLogCreate(BaseModel):
    workflow_run_id: str
    tool_name: str
    tool_group: str
    status: ToolCallStatus
    input_payload: dict[str, Any] | None = None
    output_payload: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: int | None = None


class ApprovalRequestCreate(BaseModel):
    workflow_run_id: str
    application_id: str | None = None
    action_type: str
    prompt: str
    payload: dict[str, Any] | None = None
    expires_at: datetime | None = None


class WorkflowRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    workflow_type: str
    status: WorkflowRunStatus
    current_step: str | None = None
    user_goal: str | None = None
    related_job_id: str | None = None
    related_application_id: str | None = None
    approval_request_id: str | None = None
    error: str | None = None
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


class ApprovalRequestRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    workflow_run_id: str
    application_id: str | None = None
    action_type: str
    status: ApprovalRequestStatus
    prompt: str
    payload: dict[str, Any] | None = None
    decision: str | None = None
    decided_at: datetime | None = None
    created_at: datetime
    expires_at: datetime | None = None
