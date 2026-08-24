from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.agent_runtime.loop_agent.events import LoopAgentEvent


class LoopAgentAction(str, Enum):
    CALL_TOOL = "call_tool"
    FINAL_ANSWER = "final_answer"
    WAIT_USER = "wait_user"
    REPLAN = "replan"
    STOP = "stop"


class LoopAgentStopReason(str, Enum):
    MODEL_FINAL = "model_final"
    WAITING_USER = "waiting_user"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REPLAN_REQUIRED = "replan_required"
    STOPPED = "stopped"
    STEP_FAILED = "step_failed"


@dataclass(frozen=True)
class LoopAgentDecision:
    action: LoopAgentAction
    capability: str | None = None
    tool_input: dict[str, Any] = field(default_factory=dict)
    message: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "capability": self.capability,
            "tool_input": dict(self.tool_input),
            "message": self.message,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LoopAgentObservation:
    status: str
    summary: str = ""
    result_payload: dict[str, Any] = field(default_factory=dict)
    requires_user_action: bool = False
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    suggested_next_decision: LoopAgentDecision | None = None

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "requires_user_action": self.requires_user_action,
            "tool_call_id": self.tool_call_id,
            "metadata": dict(self.metadata),
            "suggested_next_decision": self.suggested_next_decision.to_metadata_dict()
            if self.suggested_next_decision
            else None,
        }


@dataclass(frozen=True)
class LoopAgentTraceEntry:
    iteration: int
    action: LoopAgentAction
    capability: str | None = None
    decision_reason: str | None = None
    observation_status: str | None = None
    observation_summary: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "action": self.action.value,
            "capability": self.capability,
            "decision_reason": self.decision_reason,
            "observation_status": self.observation_status,
            "observation_summary": self.observation_summary,
            "tool_call_id": self.tool_call_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LoopAgentRunResult:
    stop_reason: LoopAgentStopReason
    trace: list[LoopAgentTraceEntry] = field(default_factory=list)
    final_answer: str | None = None
    pending_decision: LoopAgentDecision | None = None
    requires_user_action: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[LoopAgentEvent] = field(default_factory=list)

    @property
    def executed_step_count(self) -> int:
        return len([entry for entry in self.trace if entry.action == LoopAgentAction.CALL_TOOL])

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "control_mode": "runtime_controlled",
            "stop_reason": self.stop_reason.value,
            "executed_step_count": self.executed_step_count,
            "requires_user_action": self.requires_user_action,
            "final_answer": self.final_answer,
            "pending_decision": self.pending_decision.to_metadata_dict() if self.pending_decision else None,
            "trace": [entry.to_metadata_dict() for entry in self.trace],
            "events": [event.to_metadata_dict() for event in self.events],
            "metadata": dict(self.metadata),
        }
