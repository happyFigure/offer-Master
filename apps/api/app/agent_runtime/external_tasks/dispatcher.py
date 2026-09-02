from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent_runtime.external_tasks.executors import ExternalExecutor
from app.agent_runtime.external_tasks.schemas import (
    ApplyEntryDiscoveryResult,
    ApplyEntryDiscoveryStatus,
    ExternalTaskType,
    FindApplyEntryTaskEnvelope,
)
from app.agent_runtime.external_tasks.service import ExternalAgentTaskRepository, ExternalAgentTaskService


@dataclass(frozen=True)
class ExternalTaskDispatchResult:
    ok: bool
    task_id: str
    executor_name: str
    status: str
    result_status: str | None = None
    apply_url: str | None = None
    final_browser_url: str | None = None
    blocked_reason: str | None = None
    error: str | None = None
    next_action: str | None = None
    result_envelope: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "task_id": self.task_id,
            "executor_name": self.executor_name,
            "status": self.status,
            "result_status": self.result_status,
            "apply_url": self.apply_url,
            "final_browser_url": self.final_browser_url,
            "blocked_reason": self.blocked_reason,
            "error": self.error,
            "next_action": self.next_action,
            "result_envelope": self.result_envelope,
        }


class ExternalTaskDispatcher:
    def __init__(self, *, repository: ExternalAgentTaskRepository, executor: ExternalExecutor) -> None:
        self._repository = repository
        self._service = ExternalAgentTaskService(repository)
        self._executor = executor

    def dispatch(self, task_id: str) -> ExternalTaskDispatchResult:
        task = self._service.mark_running(task_id)
        envelope = FindApplyEntryTaskEnvelope.model_validate(task.input_payload)
        if envelope.task_type != ExternalTaskType.FIND_APPLY_ENTRY:
            raise ValueError(f"Unsupported external task type: {envelope.task_type}")

        try:
            result = self._executor.execute_find_apply_entry(envelope)
        except Exception as exc:
            result = _failed_result(envelope, self._executor.executor_name, exc)
            saved = self._service.record_result(task_id, result, executor_name=self._executor.executor_name)
            return ExternalTaskDispatchResult(
                ok=False,
                task_id=task_id,
                executor_name=self._executor.executor_name,
                status=str(saved.status.value),
                result_status=result.status.value,
                error=result.notes,
                next_action=result.next_action,
                result_envelope=_result_envelope_from_saved(saved.output_payload),
            )

        saved = self._service.record_result(task_id, result, executor_name=self._executor.executor_name)
        return ExternalTaskDispatchResult(
            ok=result.status == ApplyEntryDiscoveryStatus.FOUND_OPENED,
            task_id=task_id,
            executor_name=self._executor.executor_name,
            status=str(saved.status.value),
            result_status=result.status.value,
            apply_url=result.apply_url,
            final_browser_url=result.final_browser_url,
            blocked_reason=result.blocked_reason.value if result.blocked_reason is not None else None,
            next_action=result.next_action,
            result_envelope=_result_envelope_from_saved(saved.output_payload),
        )


def _failed_result(
    envelope: FindApplyEntryTaskEnvelope,
    executor_name: str,
    exc: Exception,
) -> ApplyEntryDiscoveryResult:
    message = f"External executor {executor_name} failed: {type(exc).__name__}: {exc}"
    return ApplyEntryDiscoveryResult(
        task_id=envelope.task_id,
        status=ApplyEntryDiscoveryStatus.FAILED,
        confidence=0,
        company_name=envelope.job.company_name,
        job_title=envelope.job.title,
        source_url=envelope.job.source_url,
        notes=message,
        next_action="retry_external_agent_dispatch",
    )


def _result_envelope_from_saved(output_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(output_payload, dict):
        return None
    envelope = output_payload.get("result_envelope")
    return dict(envelope) if isinstance(envelope, dict) else None
