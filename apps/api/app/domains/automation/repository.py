from __future__ import annotations

from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.domains.automation.models import (
    ApprovalRequest,
    AutomationRun,
    ToolCallLog,
    WorkflowCheckpoint,
    WorkflowRun,
    WorkflowRunStatus,
)


class WorkflowRunRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, workflow_run_id: str) -> WorkflowRun | None:
        return self._session.get(WorkflowRun, workflow_run_id)

    def list_active(self) -> list[WorkflowRun]:
        return list(
            self._session.scalars(
                select(WorkflowRun)
                .where(
                    WorkflowRun.status.in_(
                        [
                            WorkflowRunStatus.RUNNING,
                            WorkflowRunStatus.WAITING_USER,
                            WorkflowRunStatus.FAILED_RECOVERABLE,
                        ]
                    )
                )
                .order_by(WorkflowRun.updated_at.desc())
            ).all()
        )

    def add(self, workflow_run: WorkflowRun) -> WorkflowRun:
        self._session.add(workflow_run)
        self._session.flush()
        return workflow_run

    def flush(self) -> None:
        self._session.flush()


class WorkflowCheckpointRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, checkpoint: WorkflowCheckpoint) -> WorkflowCheckpoint:
        self._session.add(checkpoint)
        self._session.flush()
        return checkpoint


class ToolCallLogRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, tool_call_log: ToolCallLog) -> ToolCallLog:
        self._session.add(tool_call_log)
        self._session.flush()
        return tool_call_log


class ApprovalRequestRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, approval_request_id: str) -> ApprovalRequest | None:
        return self._session.get(ApprovalRequest, approval_request_id)

    def add(self, approval_request: ApprovalRequest) -> ApprovalRequest:
        self._session.add(approval_request)
        self._session.flush()
        return approval_request

    def mark_decision(
        self,
        approval_request_id: str,
        *,
        status: str,
        decision: str,
        decided_at: datetime,
    ) -> ApprovalRequest:
        self._session.execute(
            update(ApprovalRequest)
            .where(ApprovalRequest.id == approval_request_id)
            .values(status=status, decision=decision, decided_at=decided_at)
        )
        self._session.flush()
        approval = self.get(approval_request_id)
        if approval is None:
            raise ValueError(f"Approval request not found: {approval_request_id}")
        self._session.refresh(approval)
        return approval


class AutomationRunRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, automation_run: AutomationRun) -> AutomationRun:
        self._session.add(automation_run)
        self._session.flush()
        return automation_run
