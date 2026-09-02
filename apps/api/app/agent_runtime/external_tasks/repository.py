from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.agent_runtime.external_tasks.models import (
    ExternalAgentArtifact,
    ExternalAgentTask,
    ExternalAgentTaskEvent,
    utc_now,
)
from app.agent_runtime.external_tasks.schemas import ExternalAgentTaskStatus, ExternalTaskType
from app.agent_runtime.external_tasks.service import ExternalAgentTaskRecord


class SqlAlchemyExternalAgentTaskRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, task: ExternalAgentTaskRecord) -> ExternalAgentTaskRecord:
        self._session.add(
            ExternalAgentTask(
                id=task.task_id,
                task_type=_enum_value(task.task_type),
                status=_enum_value(task.status),
                trace_id=_trace_id(task.input_payload),
                context_pack_hash=task.context_pack_hash,
                input_payload=task.input_payload,
                output_payload=task.output_payload,
                blocked_reason=task.blocked_reason,
                completed_at=_completed_at(task.status),
            )
        )
        self._session.flush()
        return task

    def get(self, task_id: str) -> ExternalAgentTaskRecord | None:
        task = self._session.get(ExternalAgentTask, task_id)
        if task is None:
            return None
        return _to_record(task)

    def save(self, task: ExternalAgentTaskRecord) -> ExternalAgentTaskRecord:
        existing = self._session.get(ExternalAgentTask, task.task_id)
        if existing is None:
            return self.create(task)

        existing.task_type = _enum_value(task.task_type)
        existing.status = _enum_value(task.status)
        existing.trace_id = _trace_id(task.input_payload)
        existing.context_pack_hash = task.context_pack_hash
        existing.input_payload = task.input_payload
        existing.output_payload = task.output_payload
        existing.blocked_reason = task.blocked_reason
        existing.completed_at = existing.completed_at or _completed_at(task.status)
        self._session.flush()
        return task

    def add_event(self, task_id: str, event_type: str, payload: dict[str, Any]) -> None:
        self._session.add(
            ExternalAgentTaskEvent(
                task_id=task_id,
                event_type=event_type,
                payload=payload,
            )
        )
        self._session.flush()

    def replace_artifacts(self, task_id: str, artifacts: list[dict[str, Any]]) -> None:
        self._session.execute(
            delete(ExternalAgentArtifact).where(ExternalAgentArtifact.task_id == task_id)
        )
        for artifact in artifacts:
            self._session.add(
                ExternalAgentArtifact(
                    task_id=task_id,
                    artifact_type=str(artifact["artifact_type"]),
                    path_or_uri=str(artifact["path_or_uri"]),
                    mime_type=artifact.get("mime_type"),
                    artifact_metadata=artifact.get("metadata") or {},
                )
            )
        self._session.flush()

    def list_events(self, task_id: str) -> list[ExternalAgentTaskEvent]:
        return list(
            self._session.scalars(
                select(ExternalAgentTaskEvent)
                .where(ExternalAgentTaskEvent.task_id == task_id)
                .order_by(ExternalAgentTaskEvent.created_at)
            ).all()
        )


def _to_record(task: ExternalAgentTask) -> ExternalAgentTaskRecord:
    return ExternalAgentTaskRecord(
        task_id=task.id,
        task_type=ExternalTaskType(_enum_value(task.task_type)),
        status=ExternalAgentTaskStatus(_enum_value(task.status)),
        input_payload=task.input_payload,
        context_pack_hash=task.context_pack_hash,
        output_payload=task.output_payload,
        blocked_reason=task.blocked_reason,
    )


def _enum_value(value: str | Enum) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _trace_id(input_payload: dict[str, Any]) -> str | None:
    trace_id = input_payload.get("trace_id")
    return str(trace_id) if trace_id else None


def _completed_at(status: ExternalAgentTaskStatus | str) -> datetime | None:
    if _enum_value(status) in {
        ExternalAgentTaskStatus.SUCCEEDED.value,
        ExternalAgentTaskStatus.FAILED.value,
        ExternalAgentTaskStatus.CANCELED.value,
    }:
        return utc_now()
    return None
