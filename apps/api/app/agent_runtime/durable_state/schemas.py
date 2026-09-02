from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentTaskStatus(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class AgentStepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class AgentMemoryVisibilityScope(str, Enum):
    RUNTIME_ONLY = "runtime_only"
    MAIN_AGENT_ONLY = "main_agent_only"
    EXECUTOR_SAFE = "executor_safe"


class AgentArtifactSourceKind(str, Enum):
    RESULT_ENVELOPE = "result_envelope"
    EXTERNAL_AGENT = "external_agent"
    TOOL_RESULT = "tool_result"
    USER_UPLOAD = "user_upload"


class AgentTaskStateCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    root_workflow_run_id: str = Field(min_length=1)
    conversation_session_id: str = Field(min_length=1)
    task_type: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    status: AgentTaskStatus = AgentTaskStatus.CREATED
    current_step_id: str | None = None
    owner_executor: str | None = None
    user_goal: str | None = None
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)


class AgentStepStateCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    parent_step_id: str | None = None
    sequence_index: int = Field(ge=0)
    step_type: str = Field(min_length=1)
    status: AgentStepStatus = AgentStepStatus.PENDING
    executor_type: str = Field(min_length=1)
    executor_name: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)
    tool_call_log_id: str | None = None
    external_task_id: str | None = None
    approval_request_id: str | None = None
    retry_count: int = Field(default=0, ge=0)


class AgentMemorySnapshotCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    step_id: str | None = None
    memory_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    usage_reason: str = Field(min_length=1)
    visibility_scope: AgentMemoryVisibilityScope = AgentMemoryVisibilityScope.RUNTIME_ONLY
    passed_to_executor: bool = False
    memory_payload: dict[str, Any] = Field(default_factory=dict)


class AgentArtifactIndexCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    step_id: str | None = None
    sequence_index: int = Field(default=0, ge=0)
    source_kind: AgentArtifactSourceKind = AgentArtifactSourceKind.RESULT_ENVELOPE
    artifact_type: str = Field(min_length=1)
    title: str | None = None
    uri: str = Field(min_length=1)
    mime_type: str | None = None
    artifact_metadata: dict[str, Any] = Field(default_factory=dict)
