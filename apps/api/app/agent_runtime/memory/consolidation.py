from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent_runtime.memory.memory_candidate_extractor import (
    MemoryCandidateDraft,
    extract_memory_candidates,
)
from app.agent_runtime.memory.memory_deduplicator import deduplicate_memory_candidates
from app.agent_runtime.memory.memory_promotion import MemoryPromotionService
from app.domains.agent_memory.models import (
    AgentLearningCandidateStatus,
    AgentMemoryStatus,
    utc_now,
)
from app.domains.agent_memory.repository import AgentMemoryRepository
from app.domains.agent_memory.schemas import AgentLearningCandidateCreate
from app.domains.agent_memory.service import AgentLearningService
from app.domains.automation.models import ToolCallLog
from app.domains.conversations.models import AgentMessage, AgentMessageRole


@dataclass(frozen=True)
class MemoryConsolidationCommand:
    session_id: str | None
    workflow_run_id: str
    agent_run_id: str | None
    target_scope: str
    message_ids: list[str] = field(default_factory=list)


@dataclass
class MemoryConsolidationResult:
    workflow_run_id: str
    reviewed_message_count: int
    reviewed_tool_call_count: int
    created_candidate_ids: list[str] = field(default_factory=list)
    pending_candidate_ids: list[str] = field(default_factory=list)
    promoted_memory_ids: list[str] = field(default_factory=list)
    merged_memory_ids: list[str] = field(default_factory=list)
    skipped_reasons: list[str] = field(default_factory=list)

    @property
    def created_candidate_count(self) -> int:
        return len(self.created_candidate_ids)

    @property
    def pending_candidate_count(self) -> int:
        return len(self.pending_candidate_ids)

    @property
    def promoted_memory_count(self) -> int:
        return len(self.promoted_memory_ids)

    @property
    def merged_memory_count(self) -> int:
        return len(self.merged_memory_ids)


class MemoryConsolidationService:
    def __init__(self, *, session: Session, learning_service: AgentLearningService) -> None:
        self._session = session
        self._repository = AgentMemoryRepository(session)
        self._learning_service = learning_service
        self._promotion_service = MemoryPromotionService(self._repository)

    def consolidate(self, command: MemoryConsolidationCommand) -> MemoryConsolidationResult:
        messages = self._messages(command.session_id, message_ids=command.message_ids)
        tool_logs = self._tool_logs(command.workflow_run_id)
        drafts = extract_memory_candidates(messages=messages, tool_logs=tool_logs)
        scored_candidates = deduplicate_memory_candidates(
            drafts,
            existing_memories=self._repository.list_memories(status=AgentMemoryStatus.ACTIVE, limit=1000),
        )
        result = MemoryConsolidationResult(
            workflow_run_id=command.workflow_run_id,
            reviewed_message_count=len(messages),
            reviewed_tool_call_count=len(tool_logs),
        )

        for scored in scored_candidates:
            candidate = self._learning_service.create_learning_candidate(
                self._candidate_create(command, scored.draft, scored.score, scored.normalized_key)
            )
            result.created_candidate_ids.append(candidate.id)
            if not scored.auto_promotable:
                result.pending_candidate_ids.append(candidate.id)
                continue

            promotion = self._promotion_service.promote(
                scored,
                source_candidate_id=candidate.id,
            )
            candidate.status = AgentLearningCandidateStatus.APPLIED
            candidate.applied_at = utc_now()
            candidate.metadata_json = {
                **(candidate.metadata_json or {}),
                "promotion_target": "agent_memory",
                "promotion_mode": "automatic_low_risk",
                "promoted_memory_id": promotion.memory.id,
            }
            self._learning_service._repository.flush()
            if promotion.created:
                result.promoted_memory_ids.append(promotion.memory.id)
            else:
                result.merged_memory_ids.append(promotion.memory.id)

        return result

    def _messages(self, session_id: str | None, *, message_ids: list[str] | None = None) -> list[AgentMessage]:
        if not session_id:
            return []
        statement = (
            select(AgentMessage)
            .where(AgentMessage.session_id == session_id)
            .order_by(AgentMessage.created_at.asc(), AgentMessage.id.asc())
        )
        if message_ids:
            statement = statement.where(AgentMessage.id.in_(message_ids))
        return list(
            self._session.scalars(statement).all()
        )

    def _tool_logs(self, workflow_run_id: str) -> list[ToolCallLog]:
        return list(
            self._session.scalars(
                select(ToolCallLog)
                .where(ToolCallLog.workflow_run_id == workflow_run_id)
                .order_by(ToolCallLog.created_at.asc(), ToolCallLog.id.asc())
            ).all()
        )

    @staticmethod
    def _candidate_create(
        command: MemoryConsolidationCommand,
        draft: MemoryCandidateDraft,
        score: int,
        normalized_key: str,
    ) -> AgentLearningCandidateCreate:
        evidence_ids = list(draft.evidence_ids)
        source_message_id = evidence_ids[0] if draft.lesson_type.value == "user_preference" else None
        source_tool_call_log_id = evidence_ids[-1] if draft.lesson_type.value == "tool_recovery" else None
        target_scope = command.target_scope.strip() or draft.scope
        suggested_skill_target = f"{target_scope}-{draft.scope}"[:128]
        return AgentLearningCandidateCreate(
            source_agent_run_id=command.agent_run_id,
            source_workflow_run_id=command.workflow_run_id,
            source_tool_call_log_id=source_tool_call_log_id,
            source_message_id=source_message_id,
            lesson_type=draft.lesson_type,
            target_scope=target_scope,
            suggested_skill_target=suggested_skill_target,
            candidate_title=draft.title,
            candidate_body=draft.content,
            evidence_summary=f"Evidence ids: {', '.join(evidence_ids)}",
            success_evidence=f"deterministic_memory_score={score}",
            evidence_json={
                "evidence_ids": evidence_ids,
                "normalized_key": normalized_key,
                "memory_type": draft.memory_type,
                "memory_scope": draft.scope,
                "memory_score": score,
                "extractor_metadata": _safe_metadata(draft.metadata),
            },
            risk_level=draft.risk_level,
            metadata_json={
                "consolidation_source": "deterministic_extractor",
                "memory_normalized_key": normalized_key,
                "source_memory_type": draft.memory_type,
                "source_memory_scope": draft.scope,
            },
        )


def _safe_metadata(value: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): item
        for key, item in value.items()
        if str(key).casefold() not in {"authorization", "cookie", "password", "token", "api_key", "apikey"}
    }
