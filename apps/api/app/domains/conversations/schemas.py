from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domains.conversations.models import (
    AgentMessageKind,
    AgentMessageProvenanceKind,
    AgentMessageRole,
    AgentMessageVisibilityScope,
    AgentSessionStatus,
)


class AgentSessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    primary_intent: str | None = Field(default=None, max_length=128)
    metadata_json: dict[str, Any] | None = None


class AgentSessionUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    primary_intent: str | None = Field(default=None, max_length=128)
    metadata_json: dict[str, Any] | None = None


class AgentSessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str | None = None
    status: AgentSessionStatus
    primary_intent: str | None = None
    current_agent_run_id: str | None = None
    last_context_summary_id: str | None = None
    message_count: int
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None = None
    metadata_json: dict[str, Any] | None = None


class AgentMessageCreate(BaseModel):
    role: AgentMessageRole
    message_kind: AgentMessageKind | None = None
    agent_id: str | None = None
    recipient_agent_id: str | None = None
    visibility_scope: AgentMessageVisibilityScope | None = None
    content_text: str | None = None
    content_json: dict[str, Any] | None = None
    visible_content_text: str | None = None
    runtime_content_text: str | None = None
    content_type: str = "text/plain"
    provenance_kind: AgentMessageProvenanceKind | None = None
    agent_run_id: str | None = None
    workflow_run_id: str | None = None
    tool_call_log_id: str | None = None
    parent_message_id: str | None = None
    token_estimate: int | None = None
    exclude_from_context: bool = False
    metadata_json: dict[str, Any] | None = None


class AgentUserMessageRequest(BaseModel):
    content_text: str = Field(min_length=1)
    requested_tool_name: str | None = Field(default=None, max_length=128)
    source_type: str = Field(default="agent_chat", max_length=128)
    user_confirmed: bool = False
    tool_input: dict[str, Any] | None = None
    metadata_json: dict[str, Any] | None = None


class AgentMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    role: AgentMessageRole
    message_kind: AgentMessageKind
    agent_id: str | None = None
    recipient_agent_id: str | None = None
    visibility_scope: AgentMessageVisibilityScope
    content_text: str | None = None
    content_json: dict[str, Any] | None = None
    visible_content_text: str | None = None
    content_type: str
    provenance_kind: AgentMessageProvenanceKind | None = None
    agent_run_id: str | None = None
    workflow_run_id: str | None = None
    tool_call_log_id: str | None = None
    parent_message_id: str | None = None
    token_estimate: int | None = None
    exclude_from_context: bool
    compacted_by_summary_id: str | None = None
    created_at: datetime
    metadata_json: dict[str, Any] | None = None


class AgentContextSummaryCreate(BaseModel):
    summary_text: str
    summary_json: dict[str, Any] | None = None
    covered_message_start_id: str | None = None
    covered_message_end_id: str | None = None
    first_kept_message_id: str | None = None
    previous_summary_id: str | None = None
    token_estimate: int | None = None
    created_by: str | None = None
    metadata_json: dict[str, Any] | None = None


class AgentContextSummaryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    summary_text: str
    summary_json: dict[str, Any] | None = None
    covered_message_start_id: str | None = None
    covered_message_end_id: str | None = None
    first_kept_message_id: str | None = None
    previous_summary_id: str | None = None
    token_estimate: int | None = None
    created_at: datetime
    created_by: str | None = None
    metadata_json: dict[str, Any] | None = None


class AgentSessionListResponse(BaseModel):
    items: list[AgentSessionRead]


class AgentMessageListResponse(BaseModel):
    items: list[AgentMessageRead]


class AgentChatTurnResponse(BaseModel):
    user_message: AgentMessageRead
    assistant_message: AgentMessageRead


class AgentCompactRequest(BaseModel):
    context_window: int = Field(default=64000, ge=1)
    reserve_tokens: int = Field(default=16384, ge=0)
    keep_recent_tokens: int = Field(default=20000, ge=1)


class AgentCompactResponse(BaseModel):
    summary: AgentContextSummaryRead
    covered_message_count: int
    first_kept_message_id: str | None = None
    token_estimate_before: int
    token_estimate_after: int
    should_compact: bool
