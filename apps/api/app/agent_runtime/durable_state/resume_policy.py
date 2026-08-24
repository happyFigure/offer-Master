from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.agent_runtime.durable_state.models import AgentStepState, AgentTaskState
from app.agent_runtime.durable_state.schemas import AgentStepStatus, AgentTaskStatus


class DurableResumeAction(str, Enum):
    RETRY_FAILED_STEP = "retry_failed_step"
    CONTINUE_FROM_NEXT_STEP = "continue_from_next_step"
    WAIT_USER_ACTION = "wait_user_action"
    REISSUE_EXECUTOR_TASK = "reissue_executor_task"
    REPLAN_REMAINING_STEPS = "replan_remaining_steps"
    STOP_AND_REPORT = "stop_and_report"


@dataclass(frozen=True)
class DurableResumeDecision:
    action: DurableResumeAction
    task_id: str
    step_id: str | None = None
    reason: str | None = None
    approval_request_id: str | None = None
    executor_type: str | None = None
    executor_name: str | None = None
    capability: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DurableResumeResult:
    action: DurableResumeAction
    task_id: str
    source_step_id: str | None = None
    resume_step_id: str | None = None
    reason: str | None = None
    approval_request_id: str | None = None
    executor_type: str | None = None
    executor_name: str | None = None
    capability: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    requires_user_action: bool = False


class DurableResumePolicy:
    def __init__(self, *, max_retries: int = 2) -> None:
        self.max_retries = max(0, max_retries)

    def decide(self, task: AgentTaskState, steps: list[AgentStepState]) -> DurableResumeDecision:
        ordered_steps = sorted(steps, key=lambda step: (step.sequence_index, step.id))
        latest_step = ordered_steps[-1] if ordered_steps else None
        task_status = _status_value(task.status)

        if task_status in {AgentTaskStatus.SUCCEEDED.value, AgentTaskStatus.CANCELED.value}:
            return DurableResumeDecision(
                action=DurableResumeAction.STOP_AND_REPORT,
                task_id=task.id,
                step_id=latest_step.id if latest_step is not None else None,
                reason=f"task is already {task_status}",
            )

        if latest_step is None:
            return DurableResumeDecision(
                action=DurableResumeAction.REPLAN_REMAINING_STEPS,
                task_id=task.id,
                reason="task has no recorded steps",
            )

        step_status = _status_value(latest_step.status)
        if step_status == AgentStepStatus.WAITING_USER.value:
            return DurableResumeDecision(
                action=DurableResumeAction.WAIT_USER_ACTION,
                task_id=task.id,
                step_id=latest_step.id,
                reason="latest step is waiting for user action",
                approval_request_id=latest_step.approval_request_id,
                executor_type=latest_step.executor_type,
                executor_name=latest_step.executor_name,
                capability=latest_step.capability,
                payload=dict(latest_step.output_payload or {}),
            )

        if step_status == AgentStepStatus.FAILED.value:
            if latest_step.retry_count < self.max_retries:
                return DurableResumeDecision(
                    action=DurableResumeAction.RETRY_FAILED_STEP,
                    task_id=task.id,
                    step_id=latest_step.id,
                    reason="latest failed step is retryable",
                    executor_type=latest_step.executor_type,
                    executor_name=latest_step.executor_name,
                    capability=latest_step.capability,
                    payload=dict(latest_step.input_payload or {}),
                )
            return DurableResumeDecision(
                action=DurableResumeAction.REPLAN_REMAINING_STEPS,
                task_id=task.id,
                step_id=latest_step.id,
                reason="retry budget exhausted",
                executor_type=latest_step.executor_type,
                executor_name=latest_step.executor_name,
                capability=latest_step.capability,
            )

        if step_status == AgentStepStatus.RUNNING.value:
            return DurableResumeDecision(
                action=DurableResumeAction.REISSUE_EXECUTOR_TASK,
                task_id=task.id,
                step_id=latest_step.id,
                reason="latest step was running when interrupted",
                executor_type=latest_step.executor_type,
                executor_name=latest_step.executor_name,
                capability=latest_step.capability,
                payload=dict(latest_step.input_payload or {}),
            )

        if step_status == AgentStepStatus.SUCCEEDED.value:
            return DurableResumeDecision(
                action=DurableResumeAction.CONTINUE_FROM_NEXT_STEP,
                task_id=task.id,
                step_id=latest_step.id,
                reason="latest step succeeded",
                capability=latest_step.capability,
                payload=dict(latest_step.output_payload or {}),
            )

        return DurableResumeDecision(
            action=DurableResumeAction.REPLAN_REMAINING_STEPS,
            task_id=task.id,
            step_id=latest_step.id,
            reason=f"unsupported latest step status: {step_status}",
        )


def _status_value(value: Any) -> str:
    return str(getattr(value, "value", value))
