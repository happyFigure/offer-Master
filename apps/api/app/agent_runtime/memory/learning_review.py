from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domains.agent_memory.models import (
    AgentLearningCandidate,
    AgentLearningCandidateLessonType,
    AgentLearningCandidateRiskLevel,
)
from app.domains.agent_memory.schemas import AgentLearningCandidateCreate
from app.domains.agent_memory.service import AgentLearningService
from app.domains.automation.models import ToolCallLog, ToolCallStatus


@dataclass(frozen=True)
class LearningReviewCommand:
    agent_run_id: str | None
    session_id: str | None
    workflow_run_id: str
    target_scope: str
    suggested_skill_target: str


@dataclass(frozen=True)
class LearningReviewResult:
    workflow_run_id: str
    reviewed_tool_call_count: int
    created_count: int
    candidates: list[AgentLearningCandidate]


class LearningReviewWorkflow:
    def __init__(self, *, session: Session, learning_service: AgentLearningService) -> None:
        self._session = session
        self._learning_service = learning_service

    def review(self, command: LearningReviewCommand) -> LearningReviewResult:
        tool_logs = self._tool_logs(command.workflow_run_id)
        candidates: list[AgentLearningCandidate] = []

        for failed_log in tool_logs:
            if failed_log.status != ToolCallStatus.FAILED:
                continue
            recovered_log = self._find_recovered_log(failed_log, tool_logs)
            if recovered_log is None:
                continue
            if not self._is_learning_worthy(failed_log, recovered_log):
                continue
            candidates.append(
                self._learning_service.create_learning_candidate(
                    self._candidate_draft(command, failed_log, recovered_log)
                )
            )

        return LearningReviewResult(
            workflow_run_id=command.workflow_run_id,
            reviewed_tool_call_count=len(tool_logs),
            created_count=len(candidates),
            candidates=candidates,
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
    def _find_recovered_log(failed_log: ToolCallLog, tool_logs: list[ToolCallLog]) -> ToolCallLog | None:
        for log in tool_logs:
            if log.id == failed_log.id:
                continue
            if log.tool_name != failed_log.tool_name or log.tool_group != failed_log.tool_group:
                continue
            if log.status == ToolCallStatus.SUCCEEDED:
                return log
        return None

    @staticmethod
    def _is_learning_worthy(failed_log: ToolCallLog, recovered_log: ToolCallLog) -> bool:
        output_payload = recovered_log.output_payload or {}
        error_text = (failed_log.error or "").lower()
        if "timeout" in error_text and not output_payload.get("recovery_path"):
            return False
        if output_payload.get("verified") is True:
            return True
        if int(output_payload.get("extracted_count") or 0) > 0:
            return True
        return bool(output_payload.get("recovery_path"))

    def _candidate_draft(
        self,
        command: LearningReviewCommand,
        failed_log: ToolCallLog,
        recovered_log: ToolCallLog,
    ) -> AgentLearningCandidateCreate:
        output_payload = recovered_log.output_payload or {}
        recovery_path = self._string_value(
            output_payload.get("recovery_path"),
            default="repeat the recovered tool sequence captured in the success log",
        )
        failed_error = failed_log.error or "unknown tool failure"
        evidence_summary = (
            f"Tool {failed_log.tool_name} failed with {self._short_error(failed_error)} "
            f"and later succeeded in tool log {recovered_log.id}."
        )
        success_evidence = self._success_evidence(output_payload)

        return AgentLearningCandidateCreate(
            source_agent_run_id=command.agent_run_id,
            source_workflow_run_id=command.workflow_run_id,
            source_tool_call_log_id=recovered_log.id,
            lesson_type=AgentLearningCandidateLessonType.TOOL_RECOVERY,
            target_scope=command.target_scope,
            suggested_skill_target=command.suggested_skill_target,
            candidate_title=f"Recovered {failed_log.tool_name} tool call",
            candidate_body=(
                f"Recovery path: {recovery_path}. "
                f"Failure to avoid: {self._short_error(failed_error)}. "
                "Keep this as a review candidate; do not apply it to a skill before approval."
            ),
            evidence_summary=evidence_summary,
            success_evidence=success_evidence,
            evidence_json={
                "tool_call_log_ids": [failed_log.id, recovered_log.id],
                "failed_tool_call_log_id": failed_log.id,
                "recovered_tool_call_log_id": recovered_log.id,
                "tool_name": failed_log.tool_name,
                "tool_group": failed_log.tool_group,
                "input_payload": self._safe_payload(failed_log.input_payload),
                "output_keys": sorted(output_payload.keys()),
            },
            risk_level=self._risk_level(command.target_scope),
            metadata_json={
                "review_workflow": "hermes_style_learning_candidate",
                "session_id": command.session_id,
            },
        )

    @staticmethod
    def _short_error(error: str) -> str:
        return " ".join(error.split())[:240]

    @staticmethod
    def _success_evidence(output_payload: dict[str, Any]) -> str:
        values: list[str] = []
        if "extracted_count" in output_payload:
            values.append(f"extracted_count={output_payload['extracted_count']}")
        if output_payload.get("verified") is True:
            values.append("verified=true")
        if "recovery_path" in output_payload:
            values.append("recovery_path_present=true")
        return ", ".join(values) or "recovered tool call succeeded"

    @staticmethod
    def _safe_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
        if not payload:
            return {}
        return {
            key: value
            for key, value in payload.items()
            if key.lower() not in {"cookie", "token", "authorization", "api_key", "apikey", "password"}
        }

    @staticmethod
    def _risk_level(target_scope: str) -> AgentLearningCandidateRiskLevel:
        if target_scope in {"application_submit", "resume_submit"}:
            return AgentLearningCandidateRiskLevel.HIGH
        if target_scope in {"wechat_sync", "xiaohongshu_import", "dlmu_campus"}:
            return AgentLearningCandidateRiskLevel.MEDIUM
        return AgentLearningCandidateRiskLevel.LOW

    @staticmethod
    def _string_value(value: Any, *, default: str) -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text or default
