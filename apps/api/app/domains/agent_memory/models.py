from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def new_uuid() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class AgentMemoryStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    ARCHIVED = "archived"


class AgentLearningCandidateLessonType(str, Enum):
    TOOL_RECOVERY = "tool_recovery"
    PROVIDER_PARSER = "provider_parser"
    WORKFLOW_BOUNDARY = "workflow_boundary"
    USER_PREFERENCE = "user_preference"
    VERIFICATION_RULE = "verification_rule"


class AgentLearningCandidateRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AgentLearningCandidateStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    ARCHIVED = "archived"


class AgentSkillStorageType(str, Enum):
    MARKDOWN_FILE = "markdown_file"
    DATABASE = "database"


class AgentSkillStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    ARCHIVED = "archived"


class AgentSkillUsageEvent(str, Enum):
    USE = "use"
    VIEW = "view"
    PATCH = "patch"
    SUCCESS = "success"
    FAILURE = "failure"


class AgentMemory(Base):
    __tablename__ = "agent_memories"
    __table_args__ = (
        Index("ix_agent_memories_scope", "scope"),
        Index("ix_agent_memories_source_type", "source_type"),
        Index("ix_agent_memories_status", "status"),
        Index("ix_agent_memories_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    memory_type: Mapped[str] = mapped_column(String(64), nullable=False)
    scope: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str | None] = mapped_column(String(128))
    source_id: Mapped[str | None] = mapped_column(String(36))
    status: Mapped[AgentMemoryStatus] = mapped_column(
        String(32),
        nullable=False,
        default=AgentMemoryStatus.ACTIVE,
    )
    importance: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class AgentLearningCandidate(Base):
    __tablename__ = "agent_learning_candidates"
    __table_args__ = (
        Index("ix_agent_learning_candidates_status", "status"),
        Index("ix_agent_learning_candidates_source_workflow_run_id", "source_workflow_run_id"),
        Index("ix_agent_learning_candidates_source_tool_call_log_id", "source_tool_call_log_id"),
        Index("ix_agent_learning_candidates_target_scope", "target_scope"),
        Index("ix_agent_learning_candidates_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    source_agent_run_id: Mapped[str | None] = mapped_column(String(64))
    source_workflow_run_id: Mapped[str | None] = mapped_column(String(36))
    source_tool_call_log_id: Mapped[str | None] = mapped_column(String(36))
    source_message_id: Mapped[str | None] = mapped_column(String(36))
    lesson_type: Mapped[AgentLearningCandidateLessonType] = mapped_column(String(64), nullable=False)
    target_scope: Mapped[str] = mapped_column(String(128), nullable=False)
    suggested_skill_target: Mapped[str] = mapped_column(String(128), nullable=False)
    target_skill_id: Mapped[str | None] = mapped_column(String(36))
    candidate_title: Mapped[str] = mapped_column(String(255), nullable=False)
    candidate_body: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_summary: Mapped[str] = mapped_column(Text, nullable=False)
    success_evidence: Mapped[str | None] = mapped_column(Text)
    evidence_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    risk_level: Mapped[AgentLearningCandidateRiskLevel] = mapped_column(String(32), nullable=False)
    status: Mapped[AgentLearningCandidateStatus] = mapped_column(
        String(32),
        nullable=False,
        default=AgentLearningCandidateStatus.PENDING_REVIEW,
    )
    reviewed_by: Mapped[str | None] = mapped_column(String(64))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class AgentSkill(Base):
    __tablename__ = "agent_skills"
    __table_args__ = (
        UniqueConstraint("name", name="uq_agent_skills_name"),
        Index("ix_agent_skills_status", "status"),
        Index("ix_agent_skills_category", "category"),
        Index("ix_agent_skills_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(String(512), nullable=False)
    category: Mapped[str] = mapped_column(String(128), nullable=False)
    storage_type: Mapped[AgentSkillStorageType] = mapped_column(
        String(32),
        nullable=False,
        default=AgentSkillStorageType.MARKDOWN_FILE,
    )
    file_path: Mapped[str | None] = mapped_column(String(1024))
    status: Mapped[AgentSkillStatus] = mapped_column(
        String(32),
        nullable=False,
        default=AgentSkillStatus.ACTIVE,
    )
    protected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_by: Mapped[str] = mapped_column(String(64), nullable=False, default="developer")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    usage: Mapped[AgentSkillUsage | None] = relationship(
        back_populates="skill",
        cascade="all, delete-orphan",
        uselist=False,
    )


class AgentSkillUsage(Base):
    __tablename__ = "agent_skill_usage"
    __table_args__ = (
        UniqueConstraint("skill_id", name="uq_agent_skill_usage_skill_id"),
        Index("ix_agent_skill_usage_skill_id", "skill_id"),
        Index("ix_agent_skill_usage_state", "state"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    skill_id: Mapped[str] = mapped_column(ForeignKey("agent_skills.id"), nullable=False)
    use_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    view_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    patch_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    success_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_viewed_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_patched_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_failure_at: Mapped[datetime | None] = mapped_column(DateTime)
    state: Mapped[AgentSkillStatus] = mapped_column(
        String(32),
        nullable=False,
        default=AgentSkillStatus.ACTIVE,
    )
    archived_at: Mapped[datetime | None] = mapped_column(DateTime)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    skill: Mapped[AgentSkill] = relationship(back_populates="usage")
