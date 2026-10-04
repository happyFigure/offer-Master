from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent_runtime.state import AgentState
from app.domains.automation.models import WorkflowCheckpoint, WorkflowRun
from app.domains.automation.schemas import WorkflowCheckpointCreate
from app.domains.automation.service import AutomationService

logger = logging.getLogger(__name__)


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
        latest_before_save = self._latest_checkpoint(workflow_run_id)
        result = self._automation_service.save_checkpoint(
            WorkflowCheckpointCreate(
                workflow_run_id=workflow_run_id,
                checkpoint_key=checkpoint_key,
                state=state.to_checkpoint_state(),
            )
        )
        if latest_before_save is not None and result.checkpoint.created_at <= latest_before_save.created_at:
            # MySQL's current DATETIME column stores whole seconds. A one
            # microsecond bump is therefore truncated back to the same value,
            # which makes MAX(created_at) return several checkpoints and lets
            # an unordered LIMIT 1 resume the wrong step. Use a one-second
            # increment so the ordering survives every supported database.
            result.checkpoint.created_at = latest_before_save.created_at + timedelta(seconds=1)
            self._session.flush()
            logger.debug(
                "Adjusted checkpoint timestamp for monotonic ordering: workflow_run_id=%s checkpoint_key=%s created_at=%s",
                workflow_run_id,
                checkpoint_key,
                result.checkpoint.created_at,
            )
        return AgentCheckpointSnapshot(
            workflow_run_id=result.checkpoint.workflow_run_id,
            checkpoint_key=result.checkpoint.checkpoint_key,
            state=AgentState.from_checkpoint_state(result.checkpoint.state),
            created_at=result.checkpoint.created_at,
        )

    def load_latest(self, workflow_run_id: str) -> AgentCheckpointSnapshot:
        checkpoint = self._latest_checkpoint(workflow_run_id)
        if checkpoint is None:
            raise ValueError(f"Agent checkpoint not found: {workflow_run_id}")
        return AgentCheckpointSnapshot(
            workflow_run_id=checkpoint.workflow_run_id,
            checkpoint_key=checkpoint.checkpoint_key,
            state=AgentState.from_checkpoint_state(checkpoint.state),
            created_at=checkpoint.created_at,
        )

    def _latest_checkpoint(self, workflow_run_id: str) -> WorkflowCheckpoint | None:
        """Load the newest checkpoint without sorting the full checkpoint row set.

        The checkpoint table can become large in a long-running installation. A
        full-row ORDER BY ... LIMIT query forces MySQL to materialize and sort
        JSON state payloads, which can fail with ``Out of sort memory`` during
        approval continuation. Selecting only MAX(created_at) keeps the query
        index-friendly and the follow-up fetch returns the actual checkpoint.
        """
        latest_created_at = self._session.scalar(
            select(func.max(WorkflowCheckpoint.created_at)).where(
                WorkflowCheckpoint.workflow_run_id == workflow_run_id,
            )
        )
        if latest_created_at is None:
            return None
        candidates = select(WorkflowCheckpoint).where(
            WorkflowCheckpoint.workflow_run_id == workflow_run_id,
            WorkflowCheckpoint.created_at == latest_created_at,
        )
        workflow = self._session.get(WorkflowRun, workflow_run_id)
        current_step = str(workflow.current_step or "") if workflow is not None else ""
        if current_step:
            # Existing installations may already contain same-second rows.
            # The workflow row records the authoritative paused/completed step,
            # so use it to disambiguate those legacy timestamp collisions.
            matching_step = self._session.scalars(
                candidates.where(WorkflowCheckpoint.checkpoint_key == current_step)
                .order_by(WorkflowCheckpoint.id.desc())
                .limit(1)
            ).first()
            if matching_step is not None:
                return matching_step

        return self._session.scalars(
            candidates.order_by(WorkflowCheckpoint.id.desc()).limit(1)
        ).first()
