from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from app.agent_runtime.loop_agent.events import LoopAgentEvent
from app.agent_runtime.loop_agent.schemas import LoopAgentDecision, LoopAgentRunResult, LoopAgentStopReason


class OuterSessionStatus(str, Enum):
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass(frozen=True)
class OuterSessionRunRequest:
    session_id: str
    task_id: str
    run_id: str
    user_goal: str
    user_message: str
    is_resume: bool = False
    turn_index: int = 1
    resume_context: dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "user_goal": self.user_goal,
            "user_message": self.user_message,
            "is_resume": self.is_resume,
            "turn_index": self.turn_index,
            "resume_context": dict(self.resume_context),
        }


@dataclass
class OuterSessionState:
    session_id: str
    active_task_id: str
    user_goal: str
    status: OuterSessionStatus = OuterSessionStatus.RUNNING
    run_count: int = 0
    active_run_id: str | None = None
    waiting_message: str | None = None
    pending_decision: LoopAgentDecision | None = None
    last_inner_result: LoopAgentRunResult | None = None
    user_followups: list[str] = field(default_factory=list)
    events: list[LoopAgentEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "active_task_id": self.active_task_id,
            "user_goal": self.user_goal,
            "status": self.status.value,
            "run_count": self.run_count,
            "active_run_id": self.active_run_id,
            "waiting_message": self.waiting_message,
            "pending_decision": self.pending_decision.to_metadata_dict() if self.pending_decision else None,
            "last_inner_result": self.last_inner_result.to_metadata_dict() if self.last_inner_result else None,
            "user_followups": list(self.user_followups),
            "events": [event.to_metadata_dict() for event in self.events],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class OuterSessionTurnResult:
    status: OuterSessionStatus
    session_id: str
    task_id: str
    run_id: str
    inner_result: LoopAgentRunResult
    final_answer: str | None = None
    waiting_message: str | None = None
    requires_user_action: bool = False
    events: list[LoopAgentEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "final_answer": self.final_answer,
            "waiting_message": self.waiting_message,
            "requires_user_action": self.requires_user_action,
            "inner_result": self.inner_result.to_metadata_dict(),
            "events": [event.to_metadata_dict() for event in self.events],
            "metadata": dict(self.metadata),
        }


class InMemoryOuterSessionStore:
    def __init__(self) -> None:
        self._states: dict[str, OuterSessionState] = {}

    def get(self, session_id: str) -> OuterSessionState | None:
        return self._states.get(session_id)

    def save(self, state: OuterSessionState) -> OuterSessionState:
        self._states[state.session_id] = state
        return state


class OuterSessionLoopController:
    """MVP outer loop: keep one active task per session and resume after user input."""

    def __init__(
        self,
        *,
        run_inner_loop: Callable[[OuterSessionRunRequest], LoopAgentRunResult],
        store: InMemoryOuterSessionStore | None = None,
    ) -> None:
        self._run_inner_loop = run_inner_loop
        self._store = store or InMemoryOuterSessionStore()

    def get_state(self, session_id: str) -> OuterSessionState | None:
        return self._store.get(session_id)

    def handle_user_message(
        self,
        *,
        session_id: str,
        user_message: str,
        task_id: str | None = None,
        run_id: str | None = None,
    ) -> OuterSessionTurnResult:
        request = self.begin_user_message(
            session_id=session_id,
            user_message=user_message,
            task_id=task_id,
            run_id=run_id,
        )
        inner_result = self._run_inner_loop(request)
        return self.complete_turn(request, inner_result)

    def begin_user_message(
        self,
        *,
        session_id: str,
        user_message: str,
        task_id: str | None = None,
        run_id: str | None = None,
    ) -> OuterSessionRunRequest:
        state = self._state_for_message(session_id=session_id, user_message=user_message, task_id=task_id)
        is_resume = state.status == OuterSessionStatus.WAITING_USER
        resume_context = _resume_context(state) if is_resume else {}
        if is_resume:
            state.user_followups.append(user_message)

        state.status = OuterSessionStatus.RUNNING
        state.run_count += 1
        state.active_run_id = _non_empty(run_id) or f"outer-run-{uuid4()}"
        state.waiting_message = None
        state.pending_decision = None
        self._store.save(state)

        return OuterSessionRunRequest(
            session_id=session_id,
            task_id=state.active_task_id,
            run_id=state.active_run_id,
            user_goal=state.user_goal,
            user_message=user_message,
            is_resume=is_resume,
            turn_index=state.run_count,
            resume_context=resume_context,
        )

    def complete_turn(
        self,
        request: OuterSessionRunRequest,
        inner_result: LoopAgentRunResult,
    ) -> OuterSessionTurnResult:
        state = self._store.get(request.session_id)
        if state is None or state.active_task_id != request.task_id:
            state = OuterSessionState(
                session_id=request.session_id,
                active_task_id=request.task_id,
                user_goal=request.user_goal,
                run_count=max(0, request.turn_index - 1),
            )
        state.active_run_id = request.run_id
        if state.run_count < request.turn_index:
            state.run_count = request.turn_index
        return self._apply_inner_result(state, inner_result)

    def _state_for_message(
        self,
        *,
        session_id: str,
        user_message: str,
        task_id: str | None,
    ) -> OuterSessionState:
        current = self._store.get(session_id)
        if current is not None and current.status == OuterSessionStatus.WAITING_USER:
            return current
        if current is not None and _non_empty(task_id) and current.active_task_id == _non_empty(task_id):
            return current
        return OuterSessionState(
            session_id=session_id,
            active_task_id=_non_empty(task_id) or f"outer-task-{uuid4()}",
            user_goal=user_message,
        )

    def _apply_inner_result(
        self,
        state: OuterSessionState,
        inner_result: LoopAgentRunResult,
    ) -> OuterSessionTurnResult:
        status = _outer_status_from_inner_result(inner_result)
        waiting_message = _waiting_message(inner_result) if status == OuterSessionStatus.WAITING_USER else None
        state.status = status
        state.waiting_message = waiting_message
        state.pending_decision = inner_result.pending_decision
        state.last_inner_result = inner_result
        state.events.extend(inner_result.events)
        state.metadata = {
            **dict(state.metadata),
            "last_stop_reason": inner_result.stop_reason.value,
            "last_executed_step_count": inner_result.executed_step_count,
        }
        self._store.save(state)
        return OuterSessionTurnResult(
            status=status,
            session_id=state.session_id,
            task_id=state.active_task_id,
            run_id=state.active_run_id or "",
            inner_result=inner_result,
            final_answer=inner_result.final_answer,
            waiting_message=waiting_message,
            requires_user_action=status == OuterSessionStatus.WAITING_USER,
            events=list(inner_result.events),
            metadata={
                "run_count": state.run_count,
                "stop_reason": inner_result.stop_reason.value,
            },
        )


def _outer_status_from_inner_result(inner_result: LoopAgentRunResult) -> OuterSessionStatus:
    if inner_result.stop_reason == LoopAgentStopReason.WAITING_USER or inner_result.requires_user_action:
        return OuterSessionStatus.WAITING_USER
    if inner_result.stop_reason in {LoopAgentStopReason.MODEL_FINAL, LoopAgentStopReason.STOPPED}:
        return OuterSessionStatus.FINISHED
    return OuterSessionStatus.FAILED


def _waiting_message(inner_result: LoopAgentRunResult) -> str | None:
    decision = inner_result.pending_decision
    if decision is not None:
        return _non_empty(decision.message) or _non_empty(decision.reason)
    if inner_result.trace:
        return _non_empty(inner_result.trace[-1].observation_summary)
    return None


def _resume_context(state: OuterSessionState) -> dict[str, Any]:
    return {
        "waiting_message": state.waiting_message,
        "pending_decision": state.pending_decision.to_metadata_dict() if state.pending_decision else None,
        "previous_stop_reason": state.last_inner_result.stop_reason.value if state.last_inner_result else None,
        "previous_trace": [entry.to_metadata_dict() for entry in state.last_inner_result.trace]
        if state.last_inner_result
        else [],
    }


def _non_empty(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


__all__ = [
    "InMemoryOuterSessionStore",
    "OuterSessionLoopController",
    "OuterSessionRunRequest",
    "OuterSessionState",
    "OuterSessionStatus",
    "OuterSessionTurnResult",
]
