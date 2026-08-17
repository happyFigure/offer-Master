from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent_runtime.state import AgentState
from app.domains.automation.models import WorkflowCheckpoint
from app.domains.automation.schemas import WorkflowCheckpointCreate
from app.domains.automation.service import AutomationService


@dataclass(frozen=True)
class AgentCheckpointSnapshot:
    workflow_run_id: str
    checkpoint_key: str
    state: AgentState
    created_at: datetime


class AgentCheckpointStore:
    def __init__(self, *, session: Session, automation_service: AutomationService) -> None:
        self._session = session
        self._automation_service = automation_service

    def save(self, *, workflow_run_id: str, checkpoint_key: str, state: AgentState) -> AgentCheckpointSnapshot:
        latest_before_save = self._session.scalars(
            select(WorkflowCheckpoint)
            .where(WorkflowCheckpoint.workflow_run_id == workflow_run_id)
            .order_by(WorkflowCheckpoint.created_at.desc(), WorkflowCheckpoint.id.desc())
            .limit(1)
        ).first()
        result = self._automation_service.save_checkpoint(
            WorkflowCheckpointCreate(
                workflow_run_id=workflow_run_id,
                checkpoint_key=checkpoint_key,
                state=state.to_checkpoint_state(),
            )
        )
        if latest_before_save is not None and result.checkpoint.created_at <= latest_before_save.created_at:
            result.checkpoint.created_at = latest_before_save.created_at + timedelta(microseconds=1)
            self._session.flush()
        return AgentCheckpointSnapshot(
            workflow_run_id=result.checkpoint.workflow_run_id,
            checkpoint_key=result.checkpoint.checkpoint_key,
            state=AgentState.from_checkpoint_state(result.checkpoint.state),
            created_at=result.checkpoint.created_at,
        )

    def load_latest(self, workflow_run_id: str) -> AgentCheckpointSnapshot:
        checkpoint = self._session.scalars(
            select(WorkflowCheckpoint)
            .where(WorkflowCheckpoint.workflow_run_id == workflow_run_id)
            .order_by(WorkflowCheckpoint.created_at.desc(), WorkflowCheckpoint.id.desc())
            .limit(1)
        ).first()
        if checkpoint is None:
            raise ValueError(f"Agent checkpoint not found: {workflow_run_id}")
        return AgentCheckpointSnapshot(
            workflow_run_id=checkpoint.workflow_run_id,
            checkpoint_key=checkpoint.checkpoint_key,
            state=AgentState.from_checkpoint_state(checkpoint.state),
            created_at=checkpoint.created_at,
        )
