from __future__ import annotations

from collections.abc import Callable

from app.agent_runtime.loop_agent.events import LoopAgentEvent, LoopAgentEventType
from app.agent_runtime.loop_agent.schemas import (
    LoopAgentAction,
    LoopAgentDecision,
    LoopAgentObservation,
    LoopAgentRunResult,
    LoopAgentStopReason,
    LoopAgentTraceEntry,
)


class LoopAgentController:
    """Runtime-owned control loop for auditable multi-step agent execution."""

    def __init__(self, *, max_steps: int = 1) -> None:
        self.max_steps = max(0, max_steps)

    def run(
        self,
        *,
        decide_next_step: Callable[[list[LoopAgentTraceEntry]], LoopAgentDecision],
        execute_step: Callable[[LoopAgentDecision], LoopAgentObservation],
        session_id: str | None = None,
        task_id: str | None = None,
        run_id: str | None = None,
        event_sink: Callable[[LoopAgentEvent], None] | None = None,
    ) -> LoopAgentRunResult:
        trace: list[LoopAgentTraceEntry] = []
        events: list[LoopAgentEvent] = []
        next_decision: LoopAgentDecision | None = None

        def emit(event: LoopAgentEvent) -> None:
            events.append(event)
            if event_sink is not None:
                event_sink(event)

        def make_event(
            event_type: LoopAgentEventType,
            *,
            turn_index: int | None = None,
            step_index: int | None = None,
            capability: str | None = None,
            tool_call_id: str | None = None,
            status: str | None = None,
            summary: str | None = None,
            metadata: dict | None = None,
        ) -> LoopAgentEvent:
            return LoopAgentEvent(
                event_type=event_type,
                session_id=session_id,
                task_id=task_id,
                run_id=run_id,
                turn_index=turn_index,
                step_index=step_index,
                capability=capability,
                tool_call_id=tool_call_id,
                status=status,
                summary=summary,
                metadata=metadata or {},
            )

        def finish(result: LoopAgentRunResult, *, status: str, summary: str | None = None) -> LoopAgentRunResult:
            emit(make_event(LoopAgentEventType.TASK_FINISHED, status=status, summary=summary))
            return LoopAgentRunResult(
                stop_reason=result.stop_reason,
                trace=result.trace,
                final_answer=result.final_answer,
                pending_decision=result.pending_decision,
                requires_user_action=result.requires_user_action,
                metadata=result.metadata,
                events=events,
            )

        emit(make_event(LoopAgentEventType.TASK_STARTED, status="running"))
        for iteration in range(1, self.max_steps + 1):
            emit(make_event(LoopAgentEventType.TURN_STARTED, turn_index=iteration, status="running"))
            decision = next_decision or decide_next_step(trace)
            next_decision = None
            emit(
                make_event(
                    LoopAgentEventType.MODEL_DECISION,
                    turn_index=iteration,
                    capability=decision.capability,
                    status="decided",
                    summary=decision.reason or decision.message,
                    metadata={"decision": decision.to_metadata_dict()},
                )
            )
            if decision.action == LoopAgentAction.FINAL_ANSWER:
                return finish(
                    LoopAgentRunResult(
                        stop_reason=LoopAgentStopReason.MODEL_FINAL,
                        trace=trace,
                        final_answer=decision.message,
                        pending_decision=decision,
                    ),
                    status="succeeded",
                    summary=decision.message,
                )
            if decision.action == LoopAgentAction.WAIT_USER:
                emit(
                    make_event(
                        LoopAgentEventType.WAITING_USER,
                        turn_index=iteration,
                        capability=decision.capability,
                        status="waiting_user",
                        summary=decision.message or decision.reason,
                    )
                )
                return finish(
                    LoopAgentRunResult(
                        stop_reason=LoopAgentStopReason.WAITING_USER,
                        trace=trace,
                        pending_decision=decision,
                        requires_user_action=True,
                    ),
                    status="waiting_user",
                    summary=decision.message or decision.reason,
                )
            if decision.action == LoopAgentAction.REPLAN:
                return finish(
                    LoopAgentRunResult(
                        stop_reason=LoopAgentStopReason.REPLAN_REQUIRED,
                        trace=trace,
                        pending_decision=decision,
                    ),
                    status="replan_required",
                    summary=decision.reason,
                )
            if decision.action == LoopAgentAction.STOP:
                return finish(
                    LoopAgentRunResult(
                        stop_reason=LoopAgentStopReason.STOPPED,
                        trace=trace,
                        pending_decision=decision,
                        final_answer=decision.message,
                    ),
                    status="stopped",
                    summary=decision.message or decision.reason,
                )

            emit(
                make_event(
                    LoopAgentEventType.TOOL_STARTED,
                    turn_index=iteration,
                    step_index=iteration,
                    capability=decision.capability,
                    status="running",
                    summary=decision.reason,
                    metadata={"tool_input": decision.tool_input},
                )
            )
            observation = execute_step(decision)
            emit(
                make_event(
                    LoopAgentEventType.TOOL_FINISHED,
                    turn_index=iteration,
                    step_index=iteration,
                    capability=decision.capability,
                    tool_call_id=observation.tool_call_id,
                    status=observation.status,
                    summary=observation.summary,
                    metadata=observation.to_metadata_dict(),
                )
            )
            entry = LoopAgentTraceEntry(
                iteration=iteration,
                action=decision.action,
                capability=decision.capability,
                decision_reason=decision.reason,
                observation_status=observation.status,
                observation_summary=observation.summary,
                tool_call_id=observation.tool_call_id,
                metadata={
                    "decision": decision.metadata,
                    "observation": observation.to_metadata_dict(),
                },
            )
            trace.append(entry)
            emit(
                make_event(
                    LoopAgentEventType.TURN_FINISHED,
                    turn_index=iteration,
                    step_index=iteration,
                    capability=decision.capability,
                    tool_call_id=observation.tool_call_id,
                    status=observation.status,
                    summary=observation.summary,
                )
            )
            if observation.requires_user_action or observation.status == "waiting_user":
                emit(
                    make_event(
                        LoopAgentEventType.WAITING_USER,
                        turn_index=iteration,
                        step_index=iteration,
                        capability=decision.capability,
                        tool_call_id=observation.tool_call_id,
                        status="waiting_user",
                        summary=observation.summary,
                    )
                )
                return finish(
                    LoopAgentRunResult(
                        stop_reason=LoopAgentStopReason.WAITING_USER,
                        trace=trace,
                        pending_decision=decision,
                        requires_user_action=True,
                    ),
                    status="waiting_user",
                    summary=observation.summary,
                )
            if observation.status == "failed":
                return finish(
                    LoopAgentRunResult(
                        stop_reason=LoopAgentStopReason.STEP_FAILED,
                        trace=trace,
                        pending_decision=decision,
                    ),
                    status="failed",
                    summary=observation.summary,
                )
            next_decision = observation.suggested_next_decision

        return finish(
            LoopAgentRunResult(
                stop_reason=LoopAgentStopReason.BUDGET_EXHAUSTED,
                trace=trace,
                pending_decision=next_decision,
            ),
            status="budget_exhausted",
        )
