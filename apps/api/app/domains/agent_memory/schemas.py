from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domains.agent_memory.models import (
    AgentLearningCandidateLessonType,
    AgentLearningCandidateRiskLevel,
    AgentLearningCandidateStatus,
    AgentSkillStatus,
    AgentSkillStorageType,
    AgentSkillUsageEvent,
)


class AgentLearningCandidateCreate(BaseModel):
    source_agent_run_id: str | None = Field(default=None, max_length=64)
    source_workflow_run_id: str | None = Field(default=None, max_length=36)
    source_tool_call_log_id: str | None = Field(default=None, max_length=36)
    source_message_id: str | None = Field(default=None, max_length=36)
    lesson_type: AgentLearningCandidateLessonType
    target_scope: str = Field(min_length=1, max_length=128)
    suggested_skill_target: str = Field(min_length=1, max_length=128)
    target_skill_id: str | None = Field(default=None, max_length=36)
    candidate_title: str = Field(min_length=1, max_length=255)
    candidate_body: str = Field(min_length=1)
    evidence_summary: str = Field(min_length=1)
    success_evidence: str | None = None
    evidence_json: dict[str, Any] | None = None
    risk_level: AgentLearningCandidateRiskLevel
    metadata_json: dict[str, Any] | None = None


class AgentLearningCandidateRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source_agent_run_id: str | None = None
    source_workflow_run_id: str | None = None
    source_tool_call_log_id: str | None = None
    source_message_id: str | None = None
    lesson_type: AgentLearningCandidateLessonType
    target_scope: str
    suggested_skill_target: str
    target_skill_id: str | None = None
    candidate_title: str
    candidate_body: str
    evidence_summary: str
    success_evidence: str | None = None
    evidence_json: dict[str, Any] | None = None
    risk_level: AgentLearningCandidateRiskLevel
    status: AgentLearningCandidateStatus
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    applied_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    metadata_json: dict[str, Any] | None = None


class AgentLearningCandidateListResponse(BaseModel):
    items: list[AgentLearningCandidateRead]


class AgentSkillCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=255)
    description: str = Field(min_length=1, max_length=512)
    category: str = Field(min_length=1, max_length=128)
    protected: bool = False
    pinned: bool = False
    created_by: str = Field(default="developer", max_length=64)
    sections: dict[str, str] | None = None
    metadata_json: dict[str, Any] | None = None


class AgentSkillImportRequest(BaseModel):
    source_path: str = Field(min_length=1, max_length=2048)
    category: str = Field(default="content_source", min_length=1, max_length=128)
    protected: bool = False
    pinned: bool = False
    created_by: str = Field(default="developer", max_length=64)
    metadata_json: dict[str, Any] | None = None


class AgentSkillRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    title: str
    description: str
    category: str
    storage_type: AgentSkillStorageType
    file_path: str | None = None
    status: AgentSkillStatus
    protected: bool
    pinned: bool
    created_by: str
    created_at: datetime
    updated_at: datetime
    metadata_json: dict[str, Any] | None = None


class AgentSkillListResponse(BaseModel):
    items: list[AgentSkillRead]


class AgentSkillDocumentRead(BaseModel):
    skill: AgentSkillRead
    content: str
    version_hash: str


class AgentSkillUsageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    skill_id: str
    use_count: int
    view_count: int
    patch_count: int
    success_count: int
    failure_count: int
    last_used_at: datetime | None = None
    last_viewed_at: datetime | None = None
    last_patched_at: datetime | None = None
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    state: AgentSkillStatus
    archived_at: datetime | None = None
    metadata_json: dict[str, Any] | None = None


class AgentSkillUsageEventRequest(BaseModel):
    event: AgentSkillUsageEvent
