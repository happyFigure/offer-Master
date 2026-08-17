from __future__ import annotations

import json

from datetime import datetime

from sqlalchemy import select, text, update
from sqlalchemy.orm import Session

from app.domains.agent_memory.models import (
    AgentLearningCandidate,
    AgentLearningCandidateStatus,
    AgentMemory,
    AgentSkill,
    AgentSkillStatus,
    AgentSkillUsage,
)


class AgentMemoryRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_memory(self, memory_id: str) -> AgentMemory | None:
        return self._session.get(AgentMemory, memory_id)

    def add_memory(self, memory: AgentMemory) -> AgentMemory:
        self._session.add(memory)
        self._session.flush()
        return memory

    def get_skill(self, skill_id: str) -> AgentSkill | None:
        return self._session.get(AgentSkill, skill_id)

    def get_skill_by_name(self, name: str) -> AgentSkill | None:
        return self._session.scalar(select(AgentSkill).where(AgentSkill.name == name))

    def list_skills(
        self,
        *,
        status: AgentSkillStatus | None = None,
        limit: int = 100,
    ) -> list[AgentSkill]:
        statement = select(AgentSkill).order_by(AgentSkill.updated_at.desc(), AgentSkill.created_at.desc())
        if status is not None:
            statement = statement.where(AgentSkill.status == status)
        return list(self._session.scalars(statement.limit(limit)).all())

    def add_skill(self, skill: AgentSkill) -> AgentSkill:
        self._session.add(skill)
        self._session.flush()
        return skill

    def get_usage(self, skill_id: str) -> AgentSkillUsage | None:
        return self._session.scalar(select(AgentSkillUsage).where(AgentSkillUsage.skill_id == skill_id))

    def add_usage(self, usage: AgentSkillUsage) -> AgentSkillUsage:
        self._session.add(usage)
        self._session.flush()
        return usage

    def refresh_usage(self, usage: AgentSkillUsage) -> AgentSkillUsage:
        self._session.refresh(usage)
        return usage

    def get_usage_metadata(self, usage_id: str) -> dict | None:
        with self._session.no_autoflush:
            raw_value = self._session.execute(
                text("select metadata_json from agent_skill_usage where id = :usage_id"),
                {"usage_id": usage_id},
            ).scalar_one_or_none()
        if raw_value is None or isinstance(raw_value, dict):
            return raw_value
        if isinstance(raw_value, str):
            return json.loads(raw_value) if raw_value.strip() else None
        return dict(raw_value)

    def update_usage_runtime_event(
        self,
        usage_id: str,
        *,
        metadata_json: dict,
        recorded_at: datetime,
        success_delta: int = 0,
        failure_delta: int = 0,
    ) -> AgentSkillUsage:
        values = {
            "metadata_json": metadata_json,
            "last_used_at": recorded_at,
        }
        if success_delta:
            values["success_count"] = AgentSkillUsage.success_count + success_delta
            values["last_success_at"] = recorded_at
        if failure_delta:
            values["failure_count"] = AgentSkillUsage.failure_count + failure_delta
            values["last_failure_at"] = recorded_at
        with self._session.no_autoflush:
            self._session.execute(
                update(AgentSkillUsage)
                .where(AgentSkillUsage.id == usage_id)
                .values(**values)
                .execution_options(synchronize_session=False)
            )
        self._session.expire_all()
        usage = self._session.get(AgentSkillUsage, usage_id)
        if usage is None:
            raise ValueError(f"Agent skill usage not found: {usage_id}")
        return usage

    def get_candidate(self, candidate_id: str) -> AgentLearningCandidate | None:
        return self._session.get(AgentLearningCandidate, candidate_id)

    def list_candidates(
        self,
        *,
        status: AgentLearningCandidateStatus | None = None,
        limit: int = 100,
    ) -> list[AgentLearningCandidate]:
        statement = select(AgentLearningCandidate).order_by(
            AgentLearningCandidate.created_at.desc(),
            AgentLearningCandidate.id.desc(),
        )
        if status is not None:
            statement = statement.where(AgentLearningCandidate.status == status)
        return list(self._session.scalars(statement.limit(limit)).all())

    def add_candidate(self, candidate: AgentLearningCandidate) -> AgentLearningCandidate:
        self._session.add(candidate)
        self._session.flush()
        return candidate

    def flush(self) -> None:
        self._session.flush()
