from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent_runtime.durable_state.models import (
    AgentArtifactIndex,
    AgentMemorySnapshot,
    AgentStepState,
    AgentTaskState,
)
from app.agent_runtime.durable_state.schemas import (
    AgentArtifactIndexCreate,
    AgentMemorySnapshotCreate,
    AgentStepStateCreate,
    AgentTaskStateCreate,
)


class SqlAlchemyDurableStateRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_task(self, command: AgentTaskStateCreate) -> AgentTaskState:
        task = AgentTaskState(
            id=command.task_id,
            root_workflow_run_id=command.root_workflow_run_id,
            conversation_session_id=command.conversation_session_id,
            task_type=command.task_type,
            capability=command.capability,
            status=command.status,
            current_step_id=command.current_step_id,
            owner_executor=command.owner_executor,
            user_goal=command.user_goal,
            input_payload=command.input_payload,
            output_payload=command.output_payload,
        )
        self.session.add(task)
        self.session.flush([task])
        return task

    def get_task(self, task_id: str) -> AgentTaskState | None:
        return self.session.get(AgentTaskState, task_id)

    def update_task(self, task: AgentTaskState) -> AgentTaskState:
        self.session.add(task)
        self.session.flush([task])
        return task

    def add_step(self, command: AgentStepStateCreate) -> AgentStepState:
        step = AgentStepState(
            id=command.step_id,
            task_id=command.task_id,
            parent_step_id=command.parent_step_id,
            sequence_index=command.sequence_index,
            step_type=command.step_type,
            status=command.status,
            executor_type=command.executor_type,
            executor_name=command.executor_name,
            capability=command.capability,
            input_payload=command.input_payload,
            output_payload=command.output_payload,
            tool_call_log_id=command.tool_call_log_id,
            external_task_id=command.external_task_id,
            approval_request_id=command.approval_request_id,
            retry_count=command.retry_count,
        )
        self.session.add(step)
        self.session.flush([step])
        return step

    def get_step(self, step_id: str) -> AgentStepState | None:
        return self.session.get(AgentStepState, step_id)

    def list_steps(self, task_id: str) -> list[AgentStepState]:
        statement = (
            select(AgentStepState)
            .where(AgentStepState.task_id == task_id)
            .order_by(AgentStepState.sequence_index, AgentStepState.id)
        )
        return list(self.session.scalars(statement).all())

    def update_step(self, step: AgentStepState) -> AgentStepState:
        self.session.add(step)
        self.session.flush([step])
        return step

    def add_memory_snapshot(self, command: AgentMemorySnapshotCreate) -> AgentMemorySnapshot:
        snapshot = AgentMemorySnapshot(
            id=command.snapshot_id,
            task_id=command.task_id,
            step_id=command.step_id,
            memory_id=command.memory_id,
            source_type=command.source_type,
            usage_reason=command.usage_reason,
            visibility_scope=command.visibility_scope,
            passed_to_executor=command.passed_to_executor,
            memory_payload=command.memory_payload,
        )
        self.session.add(snapshot)
        self.session.flush([snapshot])
        return snapshot

    def list_memory_snapshots(self, task_id: str) -> list[AgentMemorySnapshot]:
        statement = (
            select(AgentMemorySnapshot)
            .where(AgentMemorySnapshot.task_id == task_id)
            .order_by(AgentMemorySnapshot.created_at, AgentMemorySnapshot.id)
        )
        return list(self.session.scalars(statement).all())

    def add_artifact(self, command: AgentArtifactIndexCreate) -> AgentArtifactIndex:
        artifact = AgentArtifactIndex(
            id=command.artifact_id,
            task_id=command.task_id,
            step_id=command.step_id,
            sequence_index=command.sequence_index,
            source_kind=command.source_kind,
            artifact_type=command.artifact_type,
            title=command.title,
            uri=command.uri,
            mime_type=command.mime_type,
            artifact_metadata=command.artifact_metadata,
        )
        self.session.add(artifact)
        self.session.flush([artifact])
        return artifact

    def list_artifacts(self, task_id: str) -> list[AgentArtifactIndex]:
        statement = (
            select(AgentArtifactIndex)
            .where(AgentArtifactIndex.task_id == task_id)
            .order_by(AgentArtifactIndex.sequence_index, AgentArtifactIndex.created_at, AgentArtifactIndex.id)
        )
        return list(self.session.scalars(statement).all())
