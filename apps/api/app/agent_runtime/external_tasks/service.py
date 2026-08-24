from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from typing import Any, Protocol

from app.agent_runtime.external_tasks.schemas import (
    ApplyEntryDiscoveryResult,
    ApplyEntryDiscoveryStatus,
    ExternalAgentTaskStatus,
    ExternalTaskType,
    FindApplyEntryTaskEnvelope,
    utc_now,
)
from app.agent_runtime.routing.result_envelope import build_apply_entry_task_result_envelope


@dataclass(frozen=True)
class ExternalAgentTaskRecord:
    task_id: str
    task_type: ExternalTaskType
    status: ExternalAgentTaskStatus
    input_payload: dict[str, Any]
    context_pack_hash: str
    output_payload: dict[str, Any] | None = None
    blocked_reason: str | None = None


class ExternalAgentTaskRepository(Protocol):
    def create(self, task: ExternalAgentTaskRecord) -> ExternalAgentTaskRecord:
        ...

    def get(self, task_id: str) -> ExternalAgentTaskRecord | None:
        ...

    def save(self, task: ExternalAgentTaskRecord) -> ExternalAgentTaskRecord:
        ...

    def add_event(self, task_id: str, event_type: str, payload: dict[str, Any]) -> None:
        ...

    def replace_artifacts(self, task_id: str, artifacts: list[dict[str, Any]]) -> None:
        ...


class ExternalAgentTaskService:
    def __init__(self, repository: ExternalAgentTaskRepository) -> None:
        self._repository = repository

    def create_find_apply_entry_task(self, envelope: FindApplyEntryTaskEnvelope) -> ExternalAgentTaskRecord:
        input_payload = envelope.model_dump(mode="json")
        task = ExternalAgentTaskRecord(
            task_id=envelope.task_id,
            task_type=envelope.task_type,
            status=ExternalAgentTaskStatus.QUEUED,
            input_payload=input_payload,
            context_pack_hash=_payload_hash(input_payload),
        )
        created = self._repository.create(task)
        self._repository.add_event(
            created.task_id,
            "task_queued",
            {"task_type": created.task_type.value, "status": created.status.value},
        )
        return created

    def mark_running(self, task_id: str) -> ExternalAgentTaskRecord:
        task = self._require_task(task_id)
        updated = replace(task, status=ExternalAgentTaskStatus.RUNNING)
        saved = self._repository.save(updated)
        self._repository.add_event(saved.task_id, "task_running", {"status": saved.status.value})
        return saved

    def record_result(
        self,
        task_id: str,
        result: ApplyEntryDiscoveryResult,
        *,
        executor_name: str | None = None,
    ) -> ExternalAgentTaskRecord:
        task = self._require_task(task_id)
        if result.task_id != task_id:
            raise ValueError(f"External task result id does not match task: {result.task_id} != {task_id}")

        status = _status_for_result(result)
        output_payload = result.model_dump(mode="json")
        output_payload["result_envelope"] = build_apply_entry_task_result_envelope(
            result_payload=output_payload,
            task_input_payload=task.input_payload,
            executor_name=executor_name,
            risk_level="medium",
        ).to_dict()
        blocked_reason = result.blocked_reason.value if result.blocked_reason is not None else None
        updated = replace(
            task,
            status=status,
            output_payload=output_payload,
            blocked_reason=blocked_reason,
        )
        saved = self._repository.save(updated)
        self._repository.replace_artifacts(
            saved.task_id,
            [artifact.model_dump(mode="json") for artifact in result.evidence_artifacts],
        )
        self._repository.add_event(
            saved.task_id,
            _event_for_status(saved.status),
            {
                "status": saved.status.value,
                "result_status": result.status.value,
                "blocked_reason": saved.blocked_reason,
                "recorded_at": utc_now().isoformat(),
                "result_envelope": output_payload["result_envelope"],
            },
        )
        return saved

    def _require_task(self, task_id: str) -> ExternalAgentTaskRecord:
        task = self._repository.get(task_id)
        if task is None:
            raise ValueError(f"External agent task not found: {task_id}")
        return task


def _status_for_result(result: ApplyEntryDiscoveryResult) -> ExternalAgentTaskStatus:
    if result.status == ApplyEntryDiscoveryStatus.FOUND_OPENED:
        return ExternalAgentTaskStatus.SUCCEEDED
    if result.status == ApplyEntryDiscoveryStatus.BLOCKED:
        return ExternalAgentTaskStatus.WAITING_USER
    return ExternalAgentTaskStatus.FAILED


def _event_for_status(status: ExternalAgentTaskStatus) -> str:
    return {
        ExternalAgentTaskStatus.SUCCEEDED: "task_succeeded",
        ExternalAgentTaskStatus.WAITING_USER: "task_waiting_user",
        ExternalAgentTaskStatus.FAILED: "task_failed",
    }.get(status, "task_updated")


def _payload_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()
