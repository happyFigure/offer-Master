from __future__ import annotations

from app.domains.agent_memory.models import (
    AgentLearningCandidate,
    AgentLearningCandidateStatus,
    AgentSkillUsageEvent,
    utc_now,
)
from app.domains.agent_memory.policies import normalize_required_text, sanitize_learning_text
from app.domains.agent_memory.repository import AgentMemoryRepository
from app.domains.agent_memory.schemas import AgentLearningCandidateCreate


class AgentLearningService:
    def __init__(self, repository: AgentMemoryRepository) -> None:
        self._repository = repository

    def create_learning_candidate(self, draft: AgentLearningCandidateCreate) -> AgentLearningCandidate:
        self._validate_candidate(draft)
        candidate = AgentLearningCandidate(
            source_agent_run_id=draft.source_agent_run_id,
            source_workflow_run_id=draft.source_workflow_run_id,
            source_tool_call_log_id=draft.source_tool_call_log_id,
            source_message_id=draft.source_message_id,
            lesson_type=draft.lesson_type,
            target_scope=normalize_required_text(draft.target_scope, "target_scope"),
            suggested_skill_target=normalize_required_text(
                draft.suggested_skill_target,
                "suggested_skill_target",
            ),
            target_skill_id=draft.target_skill_id,
            candidate_title=normalize_required_text(draft.candidate_title, "candidate_title"),
            candidate_body=sanitize_learning_text(
                normalize_required_text(draft.candidate_body, "candidate_body")
            ),
            evidence_summary=sanitize_learning_text(
                normalize_required_text(draft.evidence_summary, "evidence_summary")
            ),
            success_evidence=sanitize_learning_text(draft.success_evidence),
            evidence_json=draft.evidence_json,
            risk_level=draft.risk_level,
            status=AgentLearningCandidateStatus.PENDING_REVIEW,
            applied_at=None,
            metadata_json=draft.metadata_json,
        )
        return self._repository.add_candidate(candidate)

    def list_learning_candidates(
        self,
        *,
        status: AgentLearningCandidateStatus | None = None,
        limit: int = 100,
    ) -> list[AgentLearningCandidate]:
        return self._repository.list_candidates(status=status, limit=limit)

    def approve_candidate(self, candidate_id: str, *, reviewed_by: str = "user") -> AgentLearningCandidate:
        candidate = self._require_candidate(candidate_id)
        candidate.status = AgentLearningCandidateStatus.APPROVED
        candidate.reviewed_by = reviewed_by
        candidate.reviewed_at = utc_now()
        self._repository.flush()
        return candidate

    def reject_candidate(
        self,
        candidate_id: str,
        *,
        reviewed_by: str = "user",
        reason: str | None = None,
    ) -> AgentLearningCandidate:
        candidate = self._require_candidate(candidate_id)
        candidate.status = AgentLearningCandidateStatus.REJECTED
        candidate.reviewed_by = reviewed_by
        candidate.reviewed_at = utc_now()
        candidate.metadata_json = {**(candidate.metadata_json or {}), "reject_reason": reason}
        self._repository.flush()
        return candidate

    def apply_candidate(self, candidate_id: str, *, skill_repository=None) -> AgentLearningCandidate:
        candidate = self._require_candidate(candidate_id)
        if candidate.status != AgentLearningCandidateStatus.APPROVED:
            raise ValueError(f"Only approved learning candidates can be applied: {candidate_id}")
        if not candidate.target_skill_id:
            raise ValueError(f"Learning candidate requires target_skill_id before apply: {candidate_id}")
        if skill_repository is None:
            raise NotImplementedError("Skill repository is required to apply a learning candidate")

        patch = skill_repository.append_section(
            candidate.target_skill_id,
            heading="历史经验",
            body=self._candidate_patch_body(candidate),
            actor="agent_review",
        )
        candidate.status = AgentLearningCandidateStatus.APPLIED
        candidate.applied_at = utc_now()
        candidate.metadata_json = {
            **(candidate.metadata_json or {}),
            "applied_skill_id": candidate.target_skill_id,
            "applied_usage_event": AgentSkillUsageEvent.PATCH.value,
            "previous_skill_version_hash": patch.previous_version_hash,
            "applied_skill_version_hash": patch.applied_version_hash,
        }
        self._repository.flush()
        return candidate

    def _require_candidate(self, candidate_id: str) -> AgentLearningCandidate:
        candidate = self._repository.get_candidate(candidate_id)
        if candidate is None:
            raise ValueError(f"Agent learning candidate not found: {candidate_id}")
        return candidate

    @staticmethod
    def _validate_candidate(draft: AgentLearningCandidateCreate) -> None:
        normalize_required_text(draft.source_workflow_run_id, "source_workflow_run_id")
        normalize_required_text(draft.target_scope, "target_scope")
        normalize_required_text(draft.suggested_skill_target, "suggested_skill_target")
        normalize_required_text(draft.candidate_title, "candidate_title")
        normalize_required_text(draft.candidate_body, "candidate_body")
        normalize_required_text(draft.evidence_summary, "evidence_summary")
        evidence_ids = [
            draft.source_tool_call_log_id,
            draft.source_message_id,
            *((draft.evidence_json or {}).get("tool_call_log_ids") or []),
            *((draft.evidence_json or {}).get("message_ids") or []),
        ]
        if not any(str(value or "").strip() for value in evidence_ids):
            raise ValueError("Learning candidate requires at least one evidence id")

    @staticmethod
    def _candidate_patch_body(candidate: AgentLearningCandidate) -> str:
        lines = [
            f"#### {candidate.candidate_title}",
            "",
            candidate.candidate_body,
            "",
            f"证据：{candidate.evidence_summary}",
        ]
        if candidate.success_evidence:
            lines.extend(["", f"成功证据：{candidate.success_evidence}"])
        if candidate.source_tool_call_log_id:
            lines.extend(["", f"来源工具日志：{candidate.source_tool_call_log_id}"])
        return "\n".join(lines)
