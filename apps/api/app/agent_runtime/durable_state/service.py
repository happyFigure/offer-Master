from __future__ import annotations

from typing import Any

from uuid import uuid4

from sqlalchemy import select

from app.agent_runtime.durable_state.models import AgentArtifactIndex, AgentMemorySnapshot, AgentStepState, AgentTaskState, utc_now
from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
from app.agent_runtime.durable_state.resume_policy import DurableResumeAction, DurableResumePolicy, DurableResumeResult
from app.agent_runtime.durable_state.schemas import (
    AgentArtifactIndexCreate,
    AgentArtifactSourceKind,
    AgentMemorySnapshotCreate,
    AgentMemoryVisibilityScope,
    AgentStepStateCreate,
    AgentStepStatus,
    AgentTaskStateCreate,
    AgentTaskStatus,
)
from app.agent_runtime.external_tasks.models import ExternalAgentArtifact


class DurableStateNotFoundError(LookupError):
    pass


class DurableStateService:
    def __init__(self, repository: SqlAlchemyDurableStateRepository) -> None:
        self.repository = repository

    def create_task(
        self,
        *,
        task_id: str,
        root_workflow_run_id: str,
        conversation_session_id: str,
        task_type: str,
        capability: str,
        owner_executor: str | None = None,
        user_goal: str | None = None,
        input_payload: dict[str, Any] | None = None,
        output_payload: dict[str, Any] | None = None,
    ) -> AgentTaskState:
        return self.repository.create_task(
            AgentTaskStateCreate(
                task_id=task_id,
                root_workflow_run_id=root_workflow_run_id,
                conversation_session_id=conversation_session_id,
                task_type=task_type,
                capability=capability,
                owner_executor=owner_executor,
                user_goal=user_goal,
                input_payload=input_payload or {},
                output_payload=output_payload or {},
            )
        )

    def get_task(self, task_id: str) -> AgentTaskState:
        task = self.repository.get_task(task_id)
        if task is None:
            raise DurableStateNotFoundError(f"Agent task state not found: {task_id}")
        return task

    def add_step(
        self,
        *,
        task_id: str,
        step_id: str,
        sequence_index: int,
        step_type: str,
        executor_type: str,
        executor_name: str,
        capability: str,
        parent_step_id: str | None = None,
        input_payload: dict[str, Any] | None = None,
        output_payload: dict[str, Any] | None = None,
        status: AgentStepStatus = AgentStepStatus.PENDING,
        tool_call_log_id: str | None = None,
        external_task_id: str | None = None,
        approval_request_id: str | None = None,
        retry_count: int = 0,
    ) -> AgentStepState:
        step = self.repository.add_step(
            AgentStepStateCreate(
                step_id=step_id,
                task_id=task_id,
                parent_step_id=parent_step_id,
                sequence_index=sequence_index,
                step_type=step_type,
                status=status,
                executor_type=executor_type,
                executor_name=executor_name,
                capability=capability,
                input_payload=input_payload or {},
                output_payload=output_payload or {},
                tool_call_log_id=tool_call_log_id,
                external_task_id=external_task_id,
                approval_request_id=approval_request_id,
                retry_count=retry_count,
            )
        )
        task = self.get_task(task_id)
        task.current_step_id = step.id
        self.repository.update_task(task)
        return step

    def get_step(self, step_id: str) -> AgentStepState:
        step = self.repository.get_step(step_id)
        if step is None:
            raise DurableStateNotFoundError(f"Agent step state not found: {step_id}")
        return step

    def list_steps(self, task_id: str) -> list[AgentStepState]:
        return self.repository.list_steps(task_id)

    def mark_step_running(self, step_id: str) -> AgentStepState:
        step = self.get_step(step_id)
        step.status = AgentStepStatus.RUNNING
        if step.started_at is None:
            step.started_at = utc_now()
        self.repository.update_step(step)
        self._set_task_status(step, AgentTaskStatus.RUNNING)
        return step

    def mark_step_waiting_user(
        self,
        step_id: str,
        *,
        approval_request_id: str | None = None,
        output_payload: dict[str, Any] | None = None,
    ) -> AgentStepState:
        step = self.get_step(step_id)
        step.status = AgentStepStatus.WAITING_USER
        step.approval_request_id = approval_request_id
        if output_payload is not None:
            step.output_payload = output_payload
        self.repository.update_step(step)
        self._set_task_status(step, AgentTaskStatus.WAITING_USER)
        return step

    def mark_step_succeeded(
        self,
        step_id: str,
        *,
        tool_call_log_id: str | None = None,
        external_task_id: str | None = None,
        output_payload: dict[str, Any] | None = None,
    ) -> AgentStepState:
        step = self.get_step(step_id)
        step.status = AgentStepStatus.SUCCEEDED
        step.completed_at = utc_now()
        step.tool_call_log_id = tool_call_log_id
        step.external_task_id = external_task_id
        if output_payload is not None:
            step.output_payload = output_payload
        self.repository.update_step(step)
        self._set_task_status(step, AgentTaskStatus.RUNNING)
        return step

    def mark_step_failed(
        self,
        step_id: str,
        *,
        output_payload: dict[str, Any] | None = None,
    ) -> AgentStepState:
        step = self.get_step(step_id)
        step.status = AgentStepStatus.FAILED
        step.completed_at = utc_now()
        if output_payload is not None:
            step.output_payload = output_payload
        self.repository.update_step(step)
        self._set_task_status(step, AgentTaskStatus.FAILED, completed=True)
        return step

    def record_memory_snapshot(
        self,
        *,
        snapshot_id: str,
        task_id: str,
        memory_id: str,
        source_type: str,
        usage_reason: str,
        step_id: str | None = None,
        visibility_scope: AgentMemoryVisibilityScope = AgentMemoryVisibilityScope.RUNTIME_ONLY,
        passed_to_executor: bool = False,
        memory_payload: dict[str, Any] | None = None,
    ) -> AgentMemorySnapshot:
        self.get_task(task_id)
        if step_id is not None:
            self.get_step(step_id)
        return self.repository.add_memory_snapshot(
            AgentMemorySnapshotCreate(
                snapshot_id=snapshot_id,
                task_id=task_id,
                step_id=step_id,
                memory_id=memory_id,
                source_type=source_type,
                usage_reason=usage_reason,
                visibility_scope=visibility_scope,
                passed_to_executor=passed_to_executor,
                memory_payload=memory_payload or {},
            )
        )

    def list_memory_snapshots(self, task_id: str) -> list[AgentMemorySnapshot]:
        return self.repository.list_memory_snapshots(task_id)

    def record_artifact(
        self,
        *,
        artifact_id: str,
        task_id: str,
        artifact_type: str,
        uri: str,
        step_id: str | None = None,
        sequence_index: int = 0,
        source_kind: AgentArtifactSourceKind = AgentArtifactSourceKind.RESULT_ENVELOPE,
        title: str | None = None,
        mime_type: str | None = None,
        artifact_metadata: dict[str, Any] | None = None,
    ) -> AgentArtifactIndex:
        self.get_task(task_id)
        if step_id is not None:
            self.get_step(step_id)
        return self.repository.add_artifact(
            AgentArtifactIndexCreate(
                artifact_id=artifact_id,
                task_id=task_id,
                step_id=step_id,
                sequence_index=sequence_index,
                source_kind=source_kind,
                artifact_type=artifact_type,
                title=title,
                uri=uri,
                mime_type=mime_type,
                artifact_metadata=artifact_metadata or {},
            )
        )

    def record_artifacts_from_result_envelope(
        self,
        *,
        task_id: str,
        step_id: str | None,
        result_envelope: dict[str, Any],
    ) -> list[AgentArtifactIndex]:
        raw_artifacts = result_envelope.get("artifacts") if isinstance(result_envelope, dict) else None
        if not isinstance(raw_artifacts, list):
            return []
        records: list[AgentArtifactIndex] = []
        for index, artifact in enumerate(raw_artifacts, start=1):
            if not isinstance(artifact, dict):
                continue
            uri = _artifact_uri(artifact)
            if uri is None:
                continue
            records.append(
                self.record_artifact(
                    artifact_id=f"artifact-{uuid4()}",
                    task_id=task_id,
                    step_id=step_id,
                    sequence_index=index,
                    source_kind=AgentArtifactSourceKind.RESULT_ENVELOPE,
                    artifact_type=str(artifact.get("type") or artifact.get("artifact_type") or "artifact"),
                    title=_optional_str(artifact.get("title")) or f"artifact-{index}",
                    uri=uri,
                    mime_type=_optional_str(artifact.get("mime_type")),
                    artifact_metadata={
                        "capability": result_envelope.get("capability"),
                        "executor": result_envelope.get("executor"),
                        "raw": artifact,
                    },
                )
            )
        return records

    def list_artifacts(self, task_id: str) -> list[AgentArtifactIndex]:
        return self.repository.list_artifacts(task_id)

    def resume_task(self, task_id: str, *, max_retries: int = 2) -> DurableResumeResult:
        task = self.get_task(task_id)
        steps = self.list_steps(task_id)
        decision = DurableResumePolicy(max_retries=max_retries).decide(task, steps)

        if decision.action in {DurableResumeAction.RETRY_FAILED_STEP, DurableResumeAction.REISSUE_EXECUTOR_TASK}:
            if decision.step_id is None:
                return _resume_result_from_decision(decision)
            source_step = self.get_step(decision.step_id)
            next_index = _next_sequence_index(steps)
            resume_step = self.add_step(
                task_id=task_id,
                step_id=f"{source_step.id}:resume-{next_index}",
                parent_step_id=source_step.id,
                sequence_index=next_index,
                step_type=source_step.step_type,
                executor_type=source_step.executor_type,
                executor_name=source_step.executor_name,
                capability=source_step.capability,
                input_payload=dict(source_step.input_payload or {}),
                output_payload={},
                retry_count=source_step.retry_count + 1,
            )
            task.status = AgentTaskStatus.RUNNING
            task.current_step_id = resume_step.id
            task.completed_at = None
            self.repository.update_task(task)
            return DurableResumeResult(
                action=decision.action,
                task_id=task_id,
                source_step_id=source_step.id,
                resume_step_id=resume_step.id,
                reason=decision.reason,
                executor_type=resume_step.executor_type,
                executor_name=resume_step.executor_name,
                capability=resume_step.capability,
                payload=dict(resume_step.input_payload or {}),
            )

        if decision.action == DurableResumeAction.WAIT_USER_ACTION:
            task.status = AgentTaskStatus.WAITING_USER
            task.current_step_id = decision.step_id
            self.repository.update_task(task)
            return DurableResumeResult(
                action=decision.action,
                task_id=task_id,
                source_step_id=decision.step_id,
                reason=decision.reason,
                approval_request_id=decision.approval_request_id,
                executor_type=decision.executor_type,
                executor_name=decision.executor_name,
                capability=decision.capability,
                payload=dict(decision.payload or {}),
                requires_user_action=True,
            )

        if decision.action == DurableResumeAction.REPLAN_REMAINING_STEPS:
            task.status = AgentTaskStatus.PLANNING
            task.current_step_id = decision.step_id
            self.repository.update_task(task)

        return _resume_result_from_decision(decision)

    def sync_external_agent_artifacts(
        self,
        *,
        task_id: str,
        step_id: str | None,
        external_task_id: str,
    ) -> list[AgentArtifactIndex]:
        self.get_task(task_id)
        if step_id is not None:
            self.get_step(step_id)
        artifacts = list(
            self.repository.session.scalars(
                select(ExternalAgentArtifact)
                .where(ExternalAgentArtifact.task_id == external_task_id)
                .order_by(ExternalAgentArtifact.created_at, ExternalAgentArtifact.id)
            ).all()
        )
        records: list[AgentArtifactIndex] = []
        for index, artifact in enumerate(artifacts, start=1):
            raw_metadata = dict(artifact.artifact_metadata or {})
            records.append(
                self.record_artifact(
                    artifact_id=f"artifact-{uuid4()}",
                    task_id=task_id,
                    step_id=step_id,
                    sequence_index=index,
                    source_kind=AgentArtifactSourceKind.EXTERNAL_AGENT,
                    artifact_type=artifact.artifact_type,
                    title=_optional_str(raw_metadata.get("title")) or artifact.artifact_type,
                    uri=artifact.path_or_uri,
                    mime_type=artifact.mime_type,
                    artifact_metadata={
                        "external_task_id": external_task_id,
                        "external_artifact_id": artifact.id,
                        "raw_metadata": raw_metadata,
                    },
                )
            )
        return records

    def _set_task_status(
        self,
        step: AgentStepState,
        status: AgentTaskStatus,
        *,
        completed: bool = False,
    ) -> AgentTaskState:
        task = self.get_task(step.task_id)
        task.status = status
        task.current_step_id = step.id
        if completed:
            task.completed_at = utc_now()
        return self.repository.update_task(task)


def _artifact_uri(artifact: dict[str, Any]) -> str | None:
    for key in ("uri", "url", "path_or_uri", "path"):
        value = _optional_str(artifact.get(key))
        if value:
            return value
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _next_sequence_index(steps: list[AgentStepState]) -> int:
    if not steps:
        return 1
    return max(step.sequence_index for step in steps) + 1


def _resume_result_from_decision(decision: Any) -> DurableResumeResult:
    return DurableResumeResult(
        action=decision.action,
        task_id=decision.task_id,
        source_step_id=decision.step_id,
        reason=decision.reason,
        approval_request_id=decision.approval_request_id,
        executor_type=decision.executor_type,
        executor_name=decision.executor_name,
        capability=decision.capability,
        payload=dict(decision.payload or {}),
        requires_user_action=decision.action == DurableResumeAction.WAIT_USER_ACTION,
    )
