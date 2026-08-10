from __future__ import annotations

from dataclasses import dataclass

from app.domains.automation.events import (
    AutomationWaitingForUser,
    WorkflowCheckpointSaved,
    WorkflowStarted,
)
from app.domains.automation.models import (
    ApprovalRequest,
    ApprovalRequestStatus,
    ToolCallLog,
    WorkflowCheckpoint,
    WorkflowRun,
    WorkflowRunStatus,
    utc_now,
)
from app.domains.automation.repository import (
    ApprovalRequestRepository,
    ToolCallLogRepository,
    WorkflowCheckpointRepository,
    WorkflowRunRepository,
)
from app.domains.automation.schemas import (
    ApprovalRequestCreate,
    ToolCallLogCreate,
    WorkflowCheckpointCreate,
    WorkflowRunCreate,
)


@dataclass(frozen=True)
class WorkflowStartResult:
    workflow_run: WorkflowRun
    event: WorkflowStarted


@dataclass(frozen=True)
class ApprovalRequestResult:
    approval: ApprovalRequest
    event: AutomationWaitingForUser


@dataclass(frozen=True)
class CheckpointResult:
    checkpoint: WorkflowCheckpoint
    event: WorkflowCheckpointSaved


class AutomationService:
    def __init__(
        self,
        workflow_runs: WorkflowRunRepository,
        checkpoints: WorkflowCheckpointRepository,
        tool_call_logs: ToolCallLogRepository,
        approvals: ApprovalRequestRepository,
    ) -> None:
        self._workflow_runs = workflow_runs
        self._checkpoints = checkpoints
        self._tool_call_logs = tool_call_logs
        self._approvals = approvals

    def start_workflow(self, command: WorkflowRunCreate) -> WorkflowRun:
        workflow_run = self._workflow_runs.add(
            WorkflowRun(
                workflow_type=command.workflow_type,
                current_step=command.current_step,
                user_goal=command.user_goal,
                related_job_id=command.related_job_id,
                related_application_id=command.related_application_id,
            )
        )
        return workflow_run

    def start_workflow_with_event(self, command: WorkflowRunCreate) -> WorkflowStartResult:
        workflow_run = self.start_workflow(command)
        return WorkflowStartResult(
            workflow_run=workflow_run,
            event=WorkflowStarted(
                workflow_run_id=workflow_run.id,
                workflow_type=workflow_run.workflow_type,
                occurred_at=utc_now(),
            ),
        )

    def save_checkpoint(self, command: WorkflowCheckpointCreate) -> CheckpointResult:
        checkpoint = self._checkpoints.add(
            WorkflowCheckpoint(
                workflow_run_id=command.workflow_run_id,
                checkpoint_key=command.checkpoint_key,
                state=command.state,
            )
        )
        return CheckpointResult(
            checkpoint=checkpoint,
            event=WorkflowCheckpointSaved(
                workflow_run_id=checkpoint.workflow_run_id,
                checkpoint_key=checkpoint.checkpoint_key,
                occurred_at=utc_now(),
            ),
        )

    def record_tool_call(self, command: ToolCallLogCreate) -> ToolCallLog:
        return self._tool_call_logs.add(
            ToolCallLog(
                workflow_run_id=command.workflow_run_id,
                tool_name=command.tool_name,
                tool_group=command.tool_group,
                status=command.status,
                input_payload=command.input_payload,
                output_payload=command.output_payload,
                error=command.error,
                duration_ms=command.duration_ms,
            )
        )

    def request_user_approval(self, command: ApprovalRequestCreate) -> ApprovalRequestResult:
        workflow_run = self._workflow_runs.get(command.workflow_run_id)
        if workflow_run is None:
            raise ValueError(f"Workflow run not found: {command.workflow_run_id}")

        approval = self._approvals.add(
            ApprovalRequest(
                workflow_run=workflow_run,
                application_id=command.application_id,
                action_type=command.action_type,
                status=ApprovalRequestStatus.PENDING,
                prompt=command.prompt,
                payload=command.payload,
                expires_at=command.expires_at,
            )
        )
        workflow_run.status = WorkflowRunStatus.WAITING_USER
        workflow_run.approval_request_id = approval.id
        self._workflow_runs.flush()

        return ApprovalRequestResult(
            approval=approval,
            event=AutomationWaitingForUser(
                workflow_run_id=workflow_run.id,
                approval_request_id=approval.id,
                action_type=approval.action_type,
                occurred_at=utc_now(),
            ),
        )
