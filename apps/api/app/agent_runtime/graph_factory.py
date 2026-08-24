from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import date, datetime
from typing import Any, Callable
from uuid import uuid4

try:  # Keep LangGraph's upstream pending-deprecation noise out of targeted runtime tests.
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
except Exception:  # pragma: no cover - only relevant when langchain_core changes its warning location.
    LangChainPendingDeprecationWarning = Warning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning, module=r"langgraph\..*")

from app.agent_runtime.checkpoints import AgentCheckpointStore
from app.agent_runtime.agent_as_tool import (
    AbilityAgent,
    AgentCapabilityDefinition,
    TOOL_REGISTRY_EXECUTOR_ID,
    AgentRuntime,
    AgentRuntimeContext,
    AgentTask,
    StandardAgentResult,
    ToolRegistryAgentExecutor,
    create_default_agent_capability_registry,
)
from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolNextAction, AgentToolRuntimeGuard
from app.agent_runtime.context.capability_catalog import CapabilityCatalog
from app.agent_runtime.context.context_pack import ContextPack, ContextPackBuilder
from app.agent_runtime.durable_state.service import DurableStateNotFoundError
from app.agent_runtime.loop_agent.react_strategy import BoundedReActPolicy
from app.agent_runtime.loop_agent.schemas import (
    LoopAgentAction,
    LoopAgentDecision,
    LoopAgentObservation,
    LoopAgentStopReason,
    LoopAgentTraceEntry,
)
from app.agent_runtime.loop_agent.tool_choice_runner import LoopAgentTask, ToolChoiceLoopRunner
from app.agent_runtime.memory.compaction import CompactionConfig
from app.agent_runtime.memory.context_builder import ContextBuildConfig, MemoryContextBuilder
from app.agent_runtime.memory.skill_repository import AgentSkillRepository
from app.agent_runtime.planning.schemas import ExecutionPlan, ExecutionPlannerAction
from app.agent_runtime.reflection.capability_evaluator import CapabilityResultEvaluationRequest, CapabilityResultEvaluator
from app.agent_runtime.output_sanitizer import sanitize_agent_final_answer
from app.agent_runtime.routing.result_envelope import build_result_envelope
from app.agent_runtime.routing.runtime_guard import validate_route_decision
from app.agent_runtime.routing.schemas import RouteDecision
from app.agent_runtime.state import AgentState
from app.agent_runtime.tool_candidate_selector import ToolCandidateSelection, ToolCandidateSelector
from app.agent_runtime.tool_input import requested_sample_limit_from_text
from app.agent_runtime.tool_permissions import AgentToolPermissionPolicy
from app.agent_runtime.tool_registry import (
    APPLICATION_FIND_APPLY_ENTRY_TOOL,
    EXTERNAL_WEB_SEARCH_TOOL,
    LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
    LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
    OFFERIO_COMPANY_JOBS_TOOL,
    AgentToolDefinition,
    AgentToolRiskLevel,
    AgentToolRegistry,
)
from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
from app.agent_runtime.understanding.schemas import IntentFrame
from app.domains.automation.models import ApprovalRequest, ToolCallStatus, WorkflowRun, WorkflowRunStatus, utc_now
from app.domains.automation.schemas import ApprovalRequestCreate, ToolCallLogCreate, WorkflowRunCreate
from app.domains.automation.service import AutomationService
from app.domains.conversations.models import AgentMessageRole
from app.domains.conversations.schemas import AgentMessageCreate
from app.domains.conversations.service import ConversationService
from sqlalchemy.orm import Session


AGENT_GRAPH_NODE_ORDER = ["build_context", "plan_or_reply", "maybe_tool", "wait_confirmation", "final_response"]

# Semantic quality retry is different from transient API/network retry.
# Keep it bounded because repeated bad search rewrites can drift away from the user's intent.
DEFAULT_REFLECTION_RETRY_BUDGET = 3
MAX_REFLECTION_RETRY_BUDGET = 3


@dataclass(frozen=True)
class AgentRunCommand:
    session_id: str
    user_message: str
    requested_tool_name: str | None = None
    source_type: str = "agent_chat"
    user_confirmed: bool = False
    tool_input: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentGraphDependencies:
    automation_service: AutomationService
    checkpoint_store: AgentCheckpointStore
    conversation_service: ConversationService
    registry: AgentToolRegistry
    guard: AgentToolRuntimeGuard
    skill_repository: AgentSkillRepository | None = None
    db_session: Session | None = None
    llm_client: Any | None = None
    intent_detector: Any | None = None
    execution_planner: Any | None = None
    capability_routing_middleware: Any | None = None
    durable_state_service: Any | None = None
    agent_executors: dict[str, AbilityAgent] = field(default_factory=dict)
    capability_executor_ids: dict[str, str] = field(default_factory=dict)
    event_sink: Callable[[dict[str, Any]], None] | None = None

    def with_registry(self, registry: AgentToolRegistry) -> AgentGraphDependencies:
        return replace(self, registry=registry)

    def with_agent_runtime(
        self,
        *,
        executors: dict[str, AbilityAgent] | None = None,
        capability_executor_ids: dict[str, str] | None = None,
    ) -> AgentGraphDependencies:
        merged_executors = {**self.agent_executors, **(executors or {})}
        merged_capability_executor_ids = {**self.capability_executor_ids, **(capability_executor_ids or {})}
        for executor_id, agent in (executors or {}).items():
            for capability in _declared_agent_capabilities(agent):
                merged_capability_executor_ids.setdefault(capability.capability_id, str(executor_id))
        return replace(
            self,
            agent_executors=merged_executors,
            capability_executor_ids=merged_capability_executor_ids,
        )

    def with_event_sink(self, event_sink: Callable[[dict[str, Any]], None] | None) -> AgentGraphDependencies:
        return replace(self, event_sink=event_sink)


@dataclass(frozen=True)
class AgentWorkflowResult:
    workflow_run_id: str
    state: AgentState


@dataclass(frozen=True)
class AgentPreparedResponse:
    workflow_run_id: str
    workflow: WorkflowRun
    state: AgentState


@dataclass(frozen=True)
class AgentRuntimeGraph:
    node_order: list[str]
    compiled_graph: Any


def create_agent_graph() -> AgentRuntimeGraph:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langgraph.graph import END, StateGraph

        graph = StateGraph(dict)
        for node_name in AGENT_GRAPH_NODE_ORDER:
            graph.add_node(node_name, _identity_node)
        graph.set_entry_point("build_context")
        graph.add_edge("build_context", "plan_or_reply")
        graph.add_edge("plan_or_reply", "maybe_tool")
        graph.add_edge("maybe_tool", "wait_confirmation")
        graph.add_edge("wait_confirmation", "final_response")
        graph.add_edge("final_response", END)
        compiled_graph = graph.compile()
    return AgentRuntimeGraph(node_order=AGENT_GRAPH_NODE_ORDER.copy(), compiled_graph=compiled_graph)


def run_agent_workflow(command: AgentRunCommand, *, dependencies: AgentGraphDependencies) -> AgentWorkflowResult:
    prepared = prepare_agent_workflow_response(command, dependencies=dependencies)
    if prepared.state.current_step == "wait_confirmation":
        return AgentWorkflowResult(workflow_run_id=prepared.workflow_run_id, state=prepared.state)

    final_response, response_mode = _generate_final_response(prepared.state, dependencies=dependencies)
    return finalize_agent_workflow_response(
        prepared.state,
        final_response=final_response,
        response_mode=response_mode,
        dependencies=dependencies,
    )


def prepare_agent_workflow_response(
    command: AgentRunCommand,
    *,
    dependencies: AgentGraphDependencies,
    on_workflow_started: Callable[[WorkflowRun, AgentState], None] | None = None,
) -> AgentPreparedResponse:
    workflow = dependencies.automation_service.start_workflow(
        WorkflowRunCreate(
            workflow_type="agent_chat",
            current_step="build_context",
            user_goal=command.user_message,
        )
    )
    state = AgentState(
        session_id=command.session_id,
        workflow_run_id=workflow.id,
        agent_run_id=f"agent-run-{uuid4()}",
        user_message=command.user_message,
        current_step="build_context",
        requested_tool_name=command.requested_tool_name,
        source_type=command.source_type,
    )
    return _run_until_response_ready(
        state,
        command=command,
        workflow=workflow,
        dependencies=dependencies,
        on_workflow_started=on_workflow_started,
    )


def finalize_agent_workflow_response(
    state: AgentState,
    *,
    final_response: str,
    response_mode: str,
    dependencies: AgentGraphDependencies,
) -> AgentWorkflowResult:
    if dependencies.db_session is None:
        raise ValueError("Agent workflow finalization requires a database session.")
    workflow = dependencies.db_session.get(WorkflowRun, state.workflow_run_id)
    if workflow is None:
        raise ValueError(f"Workflow run not found: {state.workflow_run_id}")

    state, final_response, response_mode = _sanitize_final_response_for_user(
        state,
        final_response=final_response,
        response_mode=response_mode,
    )
    state = state.with_updates(
        current_step="final_response",
        final_response=final_response,
        response_mode=response_mode,
    )
    workflow.status = WorkflowRunStatus.COMPLETED
    workflow.current_step = "final_response"
    workflow.completed_at = utc_now()
    _save_step(workflow, state, dependencies)
    return AgentWorkflowResult(workflow_run_id=workflow.id, state=state)


def _sanitize_final_response_for_user(
    state: AgentState,
    *,
    final_response: str,
    response_mode: str,
) -> tuple[AgentState, str, str]:
    sanitized = sanitize_agent_final_answer(final_response)
    if not sanitized.removed_internal_protocol:
        return state, final_response, response_mode

    metadata = {
        **state.context_metadata,
        "output_sanitizer": {
            "removed_internal_protocol": sanitized.removed_internal_protocol,
            "needs_regeneration": sanitized.needs_regeneration,
            "removed_fragment_count": len(sanitized.removed_fragments),
        },
    }
    state = state.with_updates(context_metadata=metadata)
    if sanitized.content:
        return state, sanitized.content, response_mode

    tool_response = tool_result_summary_response(state)
    if tool_response is not None:
        fallback_content, fallback_mode = tool_response
        return state, fallback_content, fallback_mode
    return state, "我已完成处理，但最终回答需要重新整理。请重新发送问题或换一种问法。", "sanitized_empty_fallback"


def resume_agent_workflow(workflow_run_id: str, *, dependencies: AgentGraphDependencies) -> AgentWorkflowResult:
    snapshot = dependencies.checkpoint_store.load_latest(workflow_run_id)
    return AgentWorkflowResult(workflow_run_id=workflow_run_id, state=snapshot.state)


def continue_agent_workflow_after_approval(
    approval_request_id: str,
    *,
    approved: bool,
    decision_reason: str | None = None,
    dependencies: AgentGraphDependencies,
) -> AgentWorkflowResult:
    if dependencies.db_session is None:
        raise ValueError("Agent approval continuation requires a database session.")

    approval = dependencies.db_session.get(ApprovalRequest, approval_request_id)
    if approval is None:
        raise ValueError(f"Approval request not found: {approval_request_id}")

    approval = dependencies.automation_service.decide_approval(
        approval_request_id,
        approved=approved,
        decision=decision_reason,
    )
    snapshot = dependencies.checkpoint_store.load_latest(approval.workflow_run_id)
    workflow = dependencies.db_session.get(WorkflowRun, approval.workflow_run_id)
    if workflow is None:
        raise ValueError(f"Workflow run not found: {approval.workflow_run_id}")

    if not approved:
        state = snapshot.state.with_updates(
            current_step="approval_rejected",
            approval_request_id=approval.id,
            final_response="User rejected the pending tool call. The Agent stopped before executing the tool.",
            response_mode="user_rejected",
        )
        _record_skill_runtime_event(
            state,
            dependencies=dependencies,
            event="approval_rejected",
            evidence={
                "approval_request_id": approval.id,
                "workflow_run_id": approval.workflow_run_id,
                "agent_run_id": state.agent_run_id,
                "tool_name": approval.action_type,
                "decision_reason": decision_reason,
            },
        )
        _save_step(workflow, state, dependencies)
        return AgentWorkflowResult(workflow_run_id=workflow.id, state=state)

    if snapshot.state.current_step != "wait_confirmation":
        raise ValueError(f"Workflow is not waiting for user confirmation: {approval.workflow_run_id}")

    payload = approval.payload or {}
    requested_tool_name = str(payload.get("requested_tool_name") or approval.action_type or snapshot.state.requested_tool_name or "")
    source_type = str(payload.get("source_type") or snapshot.state.source_type or "agent_chat")
    tool_input = payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
    state = snapshot.state.with_updates(
        current_step="maybe_tool",
        approval_request_id=approval.id,
        requested_tool_name=requested_tool_name,
        source_type=source_type,
    )
    state = _maybe_tool_node(
        state,
        command=AgentRunCommand(
            session_id=state.session_id,
            user_message=state.user_message,
            requested_tool_name=requested_tool_name,
            source_type=source_type,
            user_confirmed=True,
            tool_input=tool_input,
        ),
        dependencies=dependencies,
    )
    _save_step(workflow, state, dependencies)
    state = _finalize_execution_planner_after_approval(state, dependencies=dependencies)
    state = _finalize_native_tool_loop_after_approval(state, dependencies=dependencies)
    final_response, response_mode = _generate_final_response(state, dependencies=dependencies)
    return finalize_agent_workflow_response(
        state,
        final_response=final_response,
        response_mode=response_mode,
        dependencies=dependencies,
    )


def _run_until_response_ready(
    state: AgentState,
    *,
    command: AgentRunCommand,
    workflow: WorkflowRun,
    dependencies: AgentGraphDependencies,
    on_workflow_started: Callable[[WorkflowRun, AgentState], None] | None = None,
) -> AgentPreparedResponse:
    state = _build_context_node(state, dependencies=dependencies)
    _record_durable_context_snapshots(state, dependencies=dependencies)
    _save_step(workflow, state, dependencies)
    if on_workflow_started is not None:
        on_workflow_started(workflow, state)

    route_decision: RouteDecision | None = None
    command = _auto_select_tool_command(command, state=state, registry=dependencies.registry)
    if command.requested_tool_name:
        state = state.with_updates(requested_tool_name=command.requested_tool_name, source_type=command.source_type)

    state = state.with_updates(current_step="plan_or_reply")
    _save_step(workflow, state, dependencies)

    if not command.requested_tool_name and _route_allows_tool_choice_loop(route_decision):
        state = _tool_choice_loop_node(state, dependencies=dependencies)
        if state.current_step == "wait_confirmation":
            approval = dependencies.automation_service.request_user_approval(
                ApprovalRequestCreate(
                    workflow_run_id=workflow.id,
                    action_type=state.requested_tool_name or "tool_choice_call",
                    prompt=f"Confirm before running tool: {state.requested_tool_name}",
                    payload={
                        "agent_run_id": state.agent_run_id,
                        "source_type": state.source_type,
                        "requested_tool_name": state.requested_tool_name,
                        "tool_input": _pending_runtime_tool_input(state),
                        "guard_result": state.guard_result,
                    },
                )
            ).approval
            workflow.current_step = "wait_confirmation"
            state = state.with_updates(approval_request_id=approval.id)
            _save_step(workflow, state, dependencies)
            return AgentPreparedResponse(workflow_run_id=workflow.id, workflow=workflow, state=state)
        if state.final_response and _has_prepared_final_response(state):
            _save_step(workflow, state, dependencies)
            return AgentPreparedResponse(workflow_run_id=workflow.id, workflow=workflow, state=state)

    if not command.requested_tool_name:
        route_decision = _capability_routing_node(state, dependencies=dependencies)
        if route_decision is not None:
            state = state.with_updates(
                context_metadata=_with_capability_routing_metadata(state.context_metadata, route_decision)
            )
            route_guard_result = _validate_capability_route_decision(
                route_decision,
                state=state,
                dependencies=dependencies,
            )
            state = state.with_updates(
                context_metadata={
                    **state.context_metadata,
                    "capability_routing_guard": route_guard_result,
                }
            )
            if route_guard_result.get("blocked"):
                state = state.with_updates(
                    final_response=f"Capability route blocked: {route_guard_result.get('reason')}",
                    response_mode="capability_route_blocked",
                )
            elif route_decision.route in {"ask_user", "block"}:
                state = _finalize_non_executable_route_decision(state, route_decision)
            if route_decision.capability and route_decision.route in {
                "external_agent",
                "local_tool",
                "local_workflow",
                "browser_executor",
            } and not route_guard_result.get("blocked"):
                command = replace(
                    command,
                    requested_tool_name=route_decision.capability,
                    source_type="agent_chat",
                    tool_input=dict(route_decision.tool_input),
                )
                state = state.with_updates(requested_tool_name=command.requested_tool_name, source_type=command.source_type)

    if not command.requested_tool_name and _route_allows_execution_planner(route_decision):
        state = _execution_planner_node(state, dependencies=dependencies)
        if state.current_step == "wait_confirmation":
            approval = dependencies.automation_service.request_user_approval(
                ApprovalRequestCreate(
                    workflow_run_id=workflow.id,
                    action_type=state.requested_tool_name or "planner_tool_call",
                    prompt=f"Confirm before running tool: {state.requested_tool_name}",
                    payload={
                        "agent_run_id": state.agent_run_id,
                        "source_type": state.source_type,
                        "requested_tool_name": state.requested_tool_name,
                        "tool_input": _pending_runtime_tool_input(state),
                        "guard_result": state.guard_result,
                    },
                )
            ).approval
            workflow.current_step = "wait_confirmation"
            state = state.with_updates(approval_request_id=approval.id)
            _save_step(workflow, state, dependencies)
            return AgentPreparedResponse(workflow_run_id=workflow.id, workflow=workflow, state=state)
        if state.final_response and _has_prepared_final_response(state):
            _save_step(workflow, state, dependencies)
            return AgentPreparedResponse(workflow_run_id=workflow.id, workflow=workflow, state=state)
        if state.tool_call_ids or state.requested_tool_name:
            _save_step(workflow, state, dependencies)
            return AgentPreparedResponse(workflow_run_id=workflow.id, workflow=workflow, state=state)

    if not command.requested_tool_name and _route_allows_native_tool_loop(route_decision):
        state = _native_tool_loop_node(state, dependencies=dependencies)
        if state.current_step == "wait_confirmation":
            approval = dependencies.automation_service.request_user_approval(
                ApprovalRequestCreate(
                    workflow_run_id=workflow.id,
                    action_type=state.requested_tool_name or "native_tool_call",
                    prompt=f"Confirm before running tool: {state.requested_tool_name}",
                    payload={
                        "agent_run_id": state.agent_run_id,
                        "source_type": state.source_type,
                        "requested_tool_name": state.requested_tool_name,
                        "tool_input": _pending_runtime_tool_input(state),
                        "guard_result": state.guard_result,
                    },
                )
            ).approval
            workflow.current_step = "wait_confirmation"
            state = state.with_updates(approval_request_id=approval.id)
            _save_step(workflow, state, dependencies)
            return AgentPreparedResponse(workflow_run_id=workflow.id, workflow=workflow, state=state)
        if state.final_response and _has_prepared_final_response(state):
            _save_step(workflow, state, dependencies)
            return AgentPreparedResponse(workflow_run_id=workflow.id, workflow=workflow, state=state)

    if command.requested_tool_name:
        state = _maybe_tool_node(state, command=command, dependencies=dependencies)
        if state.current_step == "wait_confirmation":
            approval = dependencies.automation_service.request_user_approval(
                ApprovalRequestCreate(
                    workflow_run_id=workflow.id,
                    action_type=command.requested_tool_name,
                    prompt=f"Confirm before running tool: {command.requested_tool_name}",
                    payload={
                        "agent_run_id": state.agent_run_id,
                        "source_type": command.source_type,
                        "requested_tool_name": command.requested_tool_name,
                        "tool_input": command.tool_input,
                        "guard_result": state.guard_result,
                    },
                )
            ).approval
            workflow.current_step = "wait_confirmation"
            state = state.with_updates(approval_request_id=approval.id)
            _record_skill_runtime_event(
                state,
                dependencies=dependencies,
                event="approval_requested",
                evidence={
                    "approval_request_id": approval.id,
                    "workflow_run_id": workflow.id,
                    "agent_run_id": state.agent_run_id,
                    "tool_name": command.requested_tool_name,
                    "source_type": command.source_type,
                    "guard_result": state.guard_result,
                },
            )
            _save_step(workflow, state, dependencies)
            return AgentPreparedResponse(workflow_run_id=workflow.id, workflow=workflow, state=state)
        _save_step(workflow, state, dependencies)

    return AgentPreparedResponse(workflow_run_id=workflow.id, workflow=workflow, state=state)


def _capability_routing_node(state: AgentState, *, dependencies: AgentGraphDependencies) -> RouteDecision | None:
    if dependencies.capability_routing_middleware is None:
        return None
    context_pack = state.context_metadata.get("context_pack") if isinstance(state.context_metadata, dict) else None
    intent_frame = state.context_metadata.get("intent_frame") if isinstance(state.context_metadata, dict) else None
    if not isinstance(context_pack, dict) or not isinstance(intent_frame, dict):
        return None
    try:
        decision = dependencies.capability_routing_middleware.decide(
            user_message=state.user_message,
            intent_frame=intent_frame,
            context_pack=context_pack,
        )
    except (AttributeError, TypeError):
        return None
    if not isinstance(decision, RouteDecision):
        return None
    return decision


def _validate_capability_route_decision(
    decision: RouteDecision,
    *,
    state: AgentState,
    dependencies: AgentGraphDependencies,
) -> dict[str, Any]:
    context_pack = state.context_metadata.get("context_pack") if isinstance(state.context_metadata, dict) else None
    if not isinstance(context_pack, dict):
        return {"ok": False, "blocked": True, "reason": "ContextPack is unavailable for capability routing."}
    return validate_route_decision(decision, context_pack=context_pack, registry=dependencies.registry).to_metadata_dict()


def _finalize_non_executable_route_decision(state: AgentState, decision: RouteDecision) -> AgentState:
    if decision.route == "ask_user":
        if decision.metadata.get("clarification_required"):
            message = str(decision.metadata.get("ask_user_message") or decision.reason or "我需要你补充一个关键信息后再继续。").strip()
            return state.with_updates(
                final_response=message,
                response_mode="clarification_ask_user",
            )
        reason = decision.reason or "this action needs explicit user confirmation"
        return state.with_updates(
            final_response=f"这个操作风险较高，需要你确认后我才能继续：{reason}",
            response_mode="capability_route_ask_user",
        )
    if decision.route == "block":
        reason = decision.reason or "this action is blocked by runtime policy"
        return state.with_updates(
            final_response=f"Capability route blocked: {reason}",
            response_mode="capability_route_blocked",
        )
    return state


def _route_allows_execution_planner(decision: RouteDecision | None) -> bool:
    if decision is None:
        return True
    return decision.route == "execution_planner"


def _route_allows_native_tool_loop(decision: RouteDecision | None) -> bool:
    if decision is None:
        return True
    return decision.route == "native_tool_loop"


def _route_allows_tool_choice_loop(decision: RouteDecision | None) -> bool:
    if decision is None:
        return True
    return decision.route == "native_tool_loop"


def _tool_choice_loop_node(state: AgentState, *, dependencies: AgentGraphDependencies) -> AgentState:
    runtime_dependencies = _dependencies_with_declared_agent_capability_tools(dependencies)
    selection = _tool_choice_loop_candidate_selection(state, dependencies=runtime_dependencies)
    capabilities = selection.capabilities
    if not capabilities:
        return state

    state_holder = {"state": state}
    _emit_tool_choice_candidate_event(runtime_dependencies, state=state, selection=selection)

    def execute_tool(_task: LoopAgentTask, decision) -> LoopAgentObservation:
        current_state = state_holder["state"]
        requested_tool_name = str(decision.capability or "")
        tool_input = dict(decision.tool_input or {})
        before_tool_call_count = len(current_state.tool_call_ids)
        next_state = _maybe_tool_node(
            current_state.with_updates(requested_tool_name=requested_tool_name, source_type="agent_chat"),
            command=AgentRunCommand(
                session_id=current_state.session_id,
                user_message=current_state.user_message,
                requested_tool_name=requested_tool_name,
                source_type="agent_chat",
                user_confirmed=False,
                tool_input=tool_input,
            ),
            dependencies=runtime_dependencies,
        )
        state_holder["state"] = next_state

        if next_state.current_step == "wait_confirmation":
            return LoopAgentObservation(
                status="waiting_user",
                summary="工具调用需要用户确认后才能继续。",
                requires_user_action=True,
                metadata={"guard_result": next_state.guard_result, "tool_input": tool_input},
            )
        if len(next_state.tool_call_ids) <= before_tool_call_count:
            guard_result = next_state.guard_result if isinstance(next_state.guard_result, dict) else {}
            return LoopAgentObservation(
                status="failed",
                summary=str(guard_result.get("user_message") or guard_result.get("reason") or "工具调用被运行时拦截。"),
                metadata={"guard_result": guard_result, "tool_input": tool_input},
            )

        payload = _latest_tool_result_payload(next_state, requested_tool_name)
        reflection = None
        if requested_tool_name == EXTERNAL_WEB_SEARCH_TOOL:
            reflection = _loop_agent_reflection_metadata(
                requested_tool_name,
                tool_input=tool_input,
                payload=payload,
                state=next_state,
                dependencies=runtime_dependencies,
                attempt_index=before_tool_call_count + 1,
            )
        observation_metadata: dict[str, Any] = {"tool_input": tool_input}
        if reflection is not None:
            observation_metadata["reflection"] = reflection
        retry_input = _reflection_retry_input_from_metadata(
            reflection,
            original_tool_input=tool_input,
            requested_tool_name=requested_tool_name,
            dependencies=runtime_dependencies,
        )
        suggested_next_decision = None
        if retry_input is not None:
            suggested_next_decision = LoopAgentDecision(
                action=LoopAgentAction.CALL_TOOL,
                capability=requested_tool_name,
                tool_input=retry_input,
                reason="工具结果不够好，运行时根据能力验收标准修改输入后重试。",
                metadata={"reflection": reflection or {}, "runtime_retry": True},
            )
        return LoopAgentObservation(
            status=_loop_agent_observation_status(payload, next_state),
            summary=_loop_agent_observation_summary(requested_tool_name, payload, next_state),
            result_payload=payload or {},
            tool_call_id=next_state.tool_call_ids[-1],
            metadata=observation_metadata,
            suggested_next_decision=suggested_next_decision,
        )

    try:
        result = ToolChoiceLoopRunner(
            registry=runtime_dependencies.registry,
            llm_client=runtime_dependencies.llm_client,
            db_session=runtime_dependencies.db_session,
            execute_tool=execute_tool,
        ).run(
            LoopAgentTask(
                user_message=state.user_message,
                available_capabilities=tuple(capabilities),
                source_type="agent_chat",
                context={
                    "mode": "model_selected_tool",
                    "candidate_selection": selection.to_metadata_dict(),
                },
            ),
            max_steps=2,
            session_id=state.session_id,
            task_id=state.workflow_run_id,
            run_id=state.agent_run_id,
            event_sink=lambda event: _emit_tool_choice_loop_event(
                runtime_dependencies,
                state=state_holder["state"],
                event=event,
            ),
        )
    except (AttributeError, TypeError):
        return state

    state = state_holder["state"]
    metadata = {
        **state.context_metadata,
        "tool_candidate_selection": selection.to_metadata_dict(),
        "tool_choice_loop": result.to_metadata_dict(),
    }
    if result.stop_reason == LoopAgentStopReason.WAITING_USER:
        return state.with_updates(context_metadata=metadata)
    if result.final_answer:
        return state.with_updates(
            final_response=result.final_answer,
            response_mode="llm_tool_choice_loop",
            context_metadata=metadata,
        )
    return state.with_updates(context_metadata=metadata)


def _emit_tool_choice_candidate_event(
    dependencies: AgentGraphDependencies,
    *,
    state: AgentState,
    selection: ToolCandidateSelection,
) -> None:
    if dependencies.event_sink is None or not selection.capabilities:
        return
    capability_labels = [format_runtime_capability_name(capability) for capability in selection.capabilities]
    dependencies.event_sink(
        {
            "event_type": "candidate_capabilities",
            "event_label": "候选能力",
            "session_id": state.session_id,
            "workflow_run_id": state.workflow_run_id,
            "agent_run_id": state.agent_run_id,
            "step_index": None,
            "tool_name": "agent_loop",
            "tool_call_id": None,
            "status": "running",
            "summary": f"主 agent 已把 {len(selection.capabilities)} 个候选能力交给模型选择：{'、'.join(capability_labels)}。",
            "tool_input_keys": [],
            "capability": None,
            "candidate_capabilities": list(selection.capabilities),
            "metadata": {"candidate_selection": selection.to_metadata_dict()},
        }
    )


def _emit_tool_choice_loop_event(
    dependencies: AgentGraphDependencies,
    *,
    state: AgentState,
    event: Any,
) -> None:
    if dependencies.event_sink is None or not hasattr(event, "to_metadata_dict"):
        return
    payload = event.to_metadata_dict()
    event_type = str(payload.get("event_type") or "loop_event")
    if event_type in {"tool_started", "tool_finished"}:
        return
    capability = str(payload.get("capability") or "") or None
    dependencies.event_sink(
        {
            "event_type": event_type,
            "event_label": str(payload.get("event_label") or _runtime_tool_event_label(event_type)),
            "session_id": str(payload.get("session_id") or state.session_id),
            "workflow_run_id": str(payload.get("task_id") or state.workflow_run_id),
            "agent_run_id": str(payload.get("run_id") or state.agent_run_id),
            "step_index": payload.get("step_index"),
            "tool_name": capability or "agent_loop",
            "tool_call_id": payload.get("tool_call_id"),
            "status": payload.get("status"),
            "summary": payload.get("summary") or _tool_choice_loop_event_summary(event_type, capability=capability),
            "tool_input_keys": [],
            "capability": capability,
            "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            "loop_event": payload,
        }
    )


def _tool_choice_loop_event_summary(event_type: str, *, capability: str | None) -> str:
    capability_label = format_runtime_capability_name(capability) if capability else "候选能力"
    return {
        "task_started": "主 agent 开始本轮工具选择循环。",
        "turn_started": "主 agent 开始判断下一步。",
        "model_decision": f"模型选择下一步使用：{capability_label}。",
        "turn_finished": f"主 agent 已观察 {capability_label} 的结果。",
        "waiting_user": "当前步骤需要用户确认或补充信息。",
        "task_finished": "本轮工具选择循环结束。",
    }.get(event_type, "Agent loop 事件已更新。")


def format_runtime_capability_name(capability: str | None) -> str:
    return {
        EXTERNAL_WEB_SEARCH_TOOL: "网页搜索",
        LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL: "本地企业库概览",
        LOCAL_JOB_SOURCE_OVERVIEW_TOOL: "岗位来源概览",
        OFFERIO_COMPANY_JOBS_TOOL: "OfferIO 岗位同步",
        APPLICATION_FIND_APPLY_ENTRY_TOOL: "申请入口发现",
        "memory_search": "会话记忆检索",
        "agent_loop": "主 agent 循环",
    }.get(str(capability or ""), str(capability or "候选能力"))


def _tool_choice_loop_available_capabilities(
    state: AgentState,
    *,
    dependencies: AgentGraphDependencies,
) -> tuple[str, ...]:
    return _tool_choice_loop_candidate_selection(state, dependencies=dependencies).capabilities


def _tool_choice_loop_candidate_selection(
    state: AgentState,
    *,
    dependencies: AgentGraphDependencies,
) -> ToolCandidateSelection:
    if dependencies.llm_client is None or dependencies.db_session is None:
        return ToolCandidateSelection()
    context_pack = state.context_metadata.get("context_pack") if isinstance(state.context_metadata, dict) else None
    allowed_capabilities = context_pack.get("allowed_capabilities") if isinstance(context_pack, dict) else None
    if isinstance(allowed_capabilities, list) and allowed_capabilities:
        return ToolCandidateSelection()
    capability_registry = create_default_agent_capability_registry(
        tool_registry=dependencies.registry,
        executor_id_by_capability=dependencies.capability_executor_ids,
    )
    return ToolCandidateSelector(capability_registry).select(state.user_message, source_type="agent_chat")


def _dependencies_with_declared_agent_capability_tools(dependencies: AgentGraphDependencies) -> AgentGraphDependencies:
    if not dependencies.agent_executors:
        return dependencies

    registry = AgentToolRegistry(dependencies.registry.list_definitions())
    existing_tool_names = set(registry.registered_tool_names())
    capability_executor_ids = dict(dependencies.capability_executor_ids)
    added = False

    for executor_id, agent in dependencies.agent_executors.items():
        for capability in _declared_agent_capabilities(agent):
            capability_executor_ids.setdefault(capability.capability_id, str(executor_id))
            if capability.capability_id in existing_tool_names:
                continue
            registry.register(_agent_capability_tool_facade(capability))
            existing_tool_names.add(capability.capability_id)
            added = True

    if not added and capability_executor_ids == dependencies.capability_executor_ids:
        return dependencies
    return replace(dependencies, registry=registry, capability_executor_ids=capability_executor_ids)


def _declared_agent_capabilities(agent: Any) -> list[AgentCapabilityDefinition]:
    capabilities = getattr(agent, "capabilities", None)
    if not callable(capabilities):
        return []
    try:
        declared = capabilities()
    except (AttributeError, TypeError):
        return []
    return [capability for capability in declared if isinstance(capability, AgentCapabilityDefinition)]


def _agent_capability_tool_facade(capability: AgentCapabilityDefinition) -> AgentToolDefinition:
    return AgentToolDefinition(
        name=capability.capability_id,
        description=capability.description,
        input_schema=dict(capability.input_schema),
        output_schema=dict(capability.output_schema),
        handler=None,
        risk_level=_agent_tool_risk_level(capability.risk_level),
        requires_confirmation=capability.requires_confirmation,
        allowed_source_types=frozenset(capability.allowed_source_types),
        result_evaluation=capability.result_evaluation,
        candidate_profile=capability.candidate_profile,
    )


def _agent_tool_risk_level(value: Any) -> AgentToolRiskLevel:
    try:
        return AgentToolRiskLevel(str(getattr(value, "value", value)))
    except ValueError:
        return AgentToolRiskLevel.MEDIUM


def _execution_planner_node(state: AgentState, *, dependencies: AgentGraphDependencies) -> AgentState:
    if dependencies.execution_planner is None or dependencies.db_session is None:
        return state

    context_pack = state.context_metadata.get("context_pack") if isinstance(state.context_metadata, dict) else None
    if not isinstance(context_pack, dict):
        return state

    try:
        plan = dependencies.execution_planner.plan(user_message=state.user_message, context_pack=context_pack)
    except (AttributeError, TypeError):
        return state
    except Exception as exc:
        return state.with_updates(
            context_metadata=_with_execution_planner_metadata(
                state.context_metadata,
                {"enabled": True, "error": str(exc), "version": "execution_planner_v1"},
            )
        )

    if not isinstance(plan, ExecutionPlan):
        return state

    state = state.with_updates(
        context_metadata={
            **_with_execution_planner_metadata(
                state.context_metadata,
                {
                    "enabled": True,
                    "version": "execution_planner_v1",
                    "mode": plan.mode,
                    "max_steps": plan.max_steps,
                },
            ),
            "execution_plan": plan.to_metadata_dict(),
        }
    )
    action = plan.primary_action()
    if action is None:
        return state
    if action.type == "final_answer":
        message = str(action.message or "").strip()
        if not message:
            return state
        return state.with_updates(final_response=message, response_mode="execution_planner")
    if action.type == "ask_user":
        message = str(action.message or "我需要你补充更多信息后才能继续。").strip()
        return state.with_updates(final_response=message, response_mode="execution_planner_ask_user")
    if action.type != "call_capability":
        return state.with_updates(
            final_response=f"当前 Execution Planner 已识别到 {action.type}，但第一版暂不执行该动作。",
            response_mode="execution_planner_unsupported",
        )
    return _execute_planner_capability_action(state, action=action, context_pack=context_pack, dependencies=dependencies)


def _execute_planner_capability_action(
    state: AgentState,
    *,
    action: ExecutionPlannerAction,
    context_pack: dict[str, Any],
    dependencies: AgentGraphDependencies,
) -> AgentState:
    capability = str(action.capability or "").strip()
    allowed_capabilities = [str(name) for name in context_pack.get("allowed_capabilities") or [] if str(name).strip()]
    if capability not in allowed_capabilities:
        return _blocked_execution_planner_response(
            state,
            reason=f"Planner requested a capability outside this turn's ContextPack: {capability}",
            details={"requested_capability": capability, "allowed_capabilities": allowed_capabilities},
        )

    definition = dependencies.registry.get(capability)
    if definition is None:
        return _blocked_execution_planner_response(
            state,
            reason=f"Planner requested an unregistered capability: {capability}",
            details={"requested_capability": capability},
        )

    tool_input = dict(action.arguments or {})
    validation_error = _validate_native_tool_input(definition.input_schema, tool_input)
    if validation_error is not None:
        return _blocked_execution_planner_response(
            state,
            reason=validation_error,
            details={"requested_capability": capability, "tool_input": tool_input},
        )

    state = state.with_updates(
        requested_tool_name=capability,
        source_type="agent_chat",
        context_metadata=_with_execution_planner_metadata(
            state.context_metadata,
            {
                "pending_action_type": action.type,
                "pending_capability": capability,
                "pending_tool_input": tool_input,
            },
        ),
    )
    state = _maybe_tool_node(
        state,
        command=AgentRunCommand(
            session_id=state.session_id,
            user_message=state.user_message,
            requested_tool_name=capability,
            source_type="agent_chat",
            user_confirmed=False,
            tool_input=tool_input,
        ),
        dependencies=dependencies,
    )
    if state.current_step == "wait_confirmation" or not state.tool_call_ids:
        return state
    return _finalize_execution_planner_after_tool(state, dependencies=dependencies)


def _finalize_execution_planner_after_tool(state: AgentState, *, dependencies: AgentGraphDependencies) -> AgentState:
    if dependencies.llm_client is None or not state.tool_call_ids:
        return state
    final_completion = dependencies.llm_client.complete(messages=state.llm_messages)
    return state.with_updates(
        final_response=final_completion.content,
        response_mode="execution_planner",
        context_metadata=_with_execution_planner_metadata(
            state.context_metadata,
            {
                "executed_capability": state.requested_tool_name,
                "tool_call_id": state.tool_call_ids[-1],
                "finalized_after_observation": True,
                "pending_tool_input": None,
            },
        ),
    )


def _finalize_execution_planner_after_approval(state: AgentState, *, dependencies: AgentGraphDependencies) -> AgentState:
    if dependencies.llm_client is None or state.current_step == "wait_confirmation" or not state.tool_call_ids:
        return state
    planner_metadata = state.context_metadata.get("execution_planner") if isinstance(state.context_metadata, dict) else None
    if not isinstance(planner_metadata, dict) or not planner_metadata.get("pending_capability"):
        return state
    return _finalize_execution_planner_after_tool(state, dependencies=dependencies)


def _blocked_execution_planner_response(state: AgentState, *, reason: str, details: dict[str, Any]) -> AgentState:
    return state.with_updates(
        final_response=f"Planner 工具调用已被拦截：{reason}",
        response_mode="tool_call_blocked",
        context_metadata=_with_execution_planner_metadata(
            state.context_metadata,
            {"enabled": True, "blocked": True, "reason": reason, "details": details},
        ),
    )


def _native_tool_loop_node(state: AgentState, *, dependencies: AgentGraphDependencies) -> AgentState:
    if dependencies.llm_client is None or dependencies.db_session is None:
        return state

    context_pack = state.context_metadata.get("context_pack") if isinstance(state.context_metadata, dict) else None
    if not isinstance(context_pack, dict):
        return state
    allowed_capabilities = [str(name) for name in context_pack.get("allowed_capabilities") or [] if str(name).strip()]
    if not allowed_capabilities:
        return state

    tool_bundle = _build_native_tool_schema_bundle(dependencies.registry, allowed_capabilities)
    if not tool_bundle["tools"]:
        return state

    requested_max_tool_calls = _native_tool_loop_max_tool_calls(context_pack)
    react_policy = BoundedReActPolicy.from_context_pack(
        context_pack,
        requested_max_steps=requested_max_tool_calls,
    )
    if not react_policy.enabled:
        return state.with_updates(
            context_metadata=_with_loop_agent_metadata(
                state.context_metadata,
                {
                    "enabled": False,
                    "version": "loop_agent_v1",
                    "strategy": "bounded_react",
                    "react_strategy": react_policy.to_metadata_dict(),
                    "disabled_reason": react_policy.disabled_reason,
                },
            )
        )

    allowed_capabilities = react_policy.allowed_capabilities
    tool_bundle = _build_native_tool_schema_bundle(dependencies.registry, allowed_capabilities)
    if not tool_bundle["tools"]:
        return state

    max_tool_calls = react_policy.max_steps
    reflection_retry_budget = _native_tool_loop_reflection_retry_budget(context_pack, allowed_capabilities)
    reflection_retry_count = 0
    loop_trace: list[LoopAgentTraceEntry] = []
    state = state.with_updates(
        context_metadata=_with_loop_agent_metadata(
            _with_tool_loop_metadata(
                state.context_metadata,
                {
                    "enabled": True,
                    "max_tool_calls": max_tool_calls,
                    "allowed_capabilities": allowed_capabilities,
                    "offered_tool_names": [tool["function"]["name"] for tool in tool_bundle["tools"]],
                },
            ),
            {
                "enabled": True,
                "version": "loop_agent_v1",
                "control_mode": "runtime_controlled",
                "strategy": "bounded_react",
                "react_strategy": react_policy.to_metadata_dict(),
                "max_steps": max_tool_calls,
                "reflection_retry_budget": reflection_retry_budget,
                "allowed_capabilities": allowed_capabilities,
                "trace": [],
            },
        )
    )
    executed_tool_names: list[str] = []
    try:
        for _iteration in range(max_tool_calls):
            completion = dependencies.llm_client.complete(
                messages=state.llm_messages,
                tools=tool_bundle["tools"],
                tool_choice="auto",
            )
            tool_calls = list(getattr(completion, "tool_calls", []) or [])
            if not tool_calls:
                content = str(getattr(completion, "content", "") or "").strip()
                if not content:
                    return state
                return state.with_updates(
                    final_response=content,
                    response_mode="llm_tool_loop",
                    context_metadata=_with_loop_agent_metadata(
                        _with_tool_loop_metadata(
                            state.context_metadata,
                            {"executed_tool_call_count": len(executed_tool_names)},
                        ),
                        _loop_agent_completion_metadata(
                            loop_trace,
                            stop_reason=LoopAgentStopReason.MODEL_FINAL,
                            final_answer=content,
                        ),
                    ),
                )

            tool_call = tool_calls[0]
            prepared = _prepare_native_tool_call(tool_call, tool_bundle, allowed_capabilities, state, dependencies)
            if isinstance(prepared, AgentState):
                return prepared
            requested_tool_name, tool_input = prepared
            state = state.with_updates(
                requested_tool_name=requested_tool_name,
                source_type="agent_chat",
                context_metadata=_with_tool_loop_metadata(
                    state.context_metadata,
                    {
                        "pending_tool_name": requested_tool_name,
                        "pending_tool_call_id": str(getattr(tool_call, "id", "") or ""),
                        "pending_tool_input": tool_input,
                    },
                ),
            )
            state = _maybe_tool_node(
                state,
                command=AgentRunCommand(
                    session_id=state.session_id,
                    user_message=state.user_message,
                    requested_tool_name=requested_tool_name,
                    source_type="agent_chat",
                    user_confirmed=False,
                    tool_input=tool_input,
                ),
                dependencies=dependencies,
            )
            if state.current_step == "wait_confirmation" or not state.tool_call_ids:
                loop_trace.append(
                    _loop_agent_trace_entry(
                        len(loop_trace) + 1,
                        requested_tool_name,
                        tool_input,
                        state,
                        dependencies=dependencies,
                    )
                )
                state = state.with_updates(
                    context_metadata=_with_loop_agent_metadata(
                        state.context_metadata,
                        _loop_agent_completion_metadata(
                            loop_trace,
                            stop_reason=LoopAgentStopReason.WAITING_USER,
                        ),
                    )
                )
                return state
            executed_tool_names.append(requested_tool_name)
            trace_entry = _loop_agent_trace_entry(
                len(loop_trace) + 1,
                requested_tool_name,
                tool_input,
                state,
                dependencies=dependencies,
            )
            loop_trace.append(trace_entry)
            state = state.with_updates(
                context_metadata=_with_loop_agent_metadata(
                    state.context_metadata,
                    {
                        **_loop_agent_progress_metadata(loop_trace),
                        "reflection_retry_count": reflection_retry_count,
                    },
                )
            )

            retry_input = _loop_agent_reflection_retry_input(
                trace_entry,
                original_tool_input=tool_input,
                requested_tool_name=requested_tool_name,
                dependencies=dependencies,
            )
            retry_source_entry = trace_entry
            while retry_input is not None and reflection_retry_count < reflection_retry_budget:
                _emit_loop_reflection_retry_event(
                    dependencies,
                    state=state,
                    requested_tool_name=requested_tool_name,
                    retry_input=retry_input,
                    trace_entry=retry_source_entry,
                )
                reflection_retry_count += 1
                state = state.with_updates(
                    requested_tool_name=requested_tool_name,
                    source_type="agent_chat",
                    context_metadata=_with_tool_loop_metadata(
                        state.context_metadata,
                        {
                            "pending_tool_name": requested_tool_name,
                            "pending_tool_call_id": f"reflection-retry-{reflection_retry_count}",
                            "pending_tool_input": retry_input,
                            "reflection_retry_count": reflection_retry_count,
                        },
                    ),
                )
                state = _maybe_tool_node(
                    state,
                    command=AgentRunCommand(
                        session_id=state.session_id,
                        user_message=state.user_message,
                        requested_tool_name=requested_tool_name,
                        source_type="agent_chat",
                        user_confirmed=False,
                        tool_input=retry_input,
                    ),
                    dependencies=dependencies,
                )
                if state.current_step == "wait_confirmation" or not state.tool_call_ids:
                    loop_trace.append(
                        _loop_agent_trace_entry(
                            len(loop_trace) + 1,
                            requested_tool_name,
                            retry_input,
                            state,
                            dependencies=dependencies,
                        )
                    )
                    state = state.with_updates(
                        context_metadata=_with_loop_agent_metadata(
                            state.context_metadata,
                            _loop_agent_completion_metadata(
                                loop_trace,
                                stop_reason=LoopAgentStopReason.WAITING_USER,
                            ),
                        )
                    )
                    return state
                executed_tool_names.append(requested_tool_name)
                retry_trace_entry = _loop_agent_trace_entry(
                    len(loop_trace) + 1,
                    requested_tool_name,
                    retry_input,
                    state,
                    dependencies=dependencies,
                )
                loop_trace.append(retry_trace_entry)
                state = state.with_updates(
                    context_metadata=_with_loop_agent_metadata(
                        state.context_metadata,
                        {
                            **_loop_agent_progress_metadata(loop_trace),
                            "reflection_retry_count": reflection_retry_count,
                        },
                    )
                )
                retry_input = _loop_agent_reflection_retry_input(
                    retry_trace_entry,
                    original_tool_input=retry_input,
                    requested_tool_name=requested_tool_name,
                    dependencies=dependencies,
                )
                retry_source_entry = retry_trace_entry
    except (AttributeError, TypeError):
        return state

    final_completion, stop_reason = _finalize_native_tool_loop_completion(
        dependencies.llm_client,
        messages=state.llm_messages,
        tools=tool_bundle["tools"] if max_tool_calls > 1 else None,
    )
    return state.with_updates(
        final_response=final_completion.content,
        response_mode="llm_tool_loop",
        context_metadata=_with_loop_agent_metadata(
            _with_tool_loop_metadata(
                state.context_metadata,
                {
                    "enabled": True,
                    "executed_tool_name": executed_tool_names[-1] if executed_tool_names else None,
                    "executed_tool_names": executed_tool_names,
                    "executed_tool_call_count": len(executed_tool_names),
                    "tool_call_id": state.tool_call_ids[-1],
                    "finalized_after_observation": True,
                },
            ),
            _loop_agent_completion_metadata(
                loop_trace,
                stop_reason=stop_reason,
                final_answer=final_completion.content,
            ),
        ),
    )


def _native_tool_loop_max_tool_calls(context_pack: dict[str, Any]) -> int:
    intent_frame = context_pack.get("intent_frame") if isinstance(context_pack.get("intent_frame"), dict) else None
    entities = intent_frame.get("entities") if isinstance(intent_frame, dict) else context_pack.get("entities")
    company_names = entities.get("company_names") if isinstance(entities, dict) else []
    if isinstance(company_names, list) and len(company_names) > 1:
        return min(5, max(2, len(company_names)))
    return 1


def _native_tool_loop_reflection_retry_budget(context_pack: dict[str, Any], allowed_capabilities: list[str]) -> int:
    if EXTERNAL_WEB_SEARCH_TOOL not in allowed_capabilities:
        return 0
    configured = _reflection_retry_budget_config_value(context_pack)
    if configured is None:
        return DEFAULT_REFLECTION_RETRY_BUDGET
    try:
        budget = int(configured)
    except (TypeError, ValueError):
        return DEFAULT_REFLECTION_RETRY_BUDGET
    return max(0, min(MAX_REFLECTION_RETRY_BUDGET, budget))


def _reflection_retry_budget_config_value(context_pack: dict[str, Any]) -> Any | None:
    loop_agent_config = context_pack.get("loop_agent") if isinstance(context_pack.get("loop_agent"), dict) else None
    if isinstance(loop_agent_config, dict) and loop_agent_config.get("reflection_retry_budget") is not None:
        return loop_agent_config.get("reflection_retry_budget")
    if context_pack.get("reflection_retry_budget") is not None:
        return context_pack.get("reflection_retry_budget")
    return None


def _prepare_native_tool_call(
    tool_call: Any,
    tool_bundle: dict[str, Any],
    allowed_capabilities: list[str],
    state: AgentState,
    dependencies: AgentGraphDependencies,
) -> tuple[str, dict[str, Any]] | AgentState:
    requested_tool_name = tool_bundle["alias_to_tool_name"].get(str(tool_call.name), str(tool_call.name))
    if requested_tool_name not in allowed_capabilities:
        return _blocked_native_tool_call_response(
            state,
            reason=f"Model requested a tool outside this turn's ContextPack: {requested_tool_name}",
            details={"requested_tool_name": requested_tool_name, "allowed_capabilities": allowed_capabilities},
        )

    definition = dependencies.registry.get(requested_tool_name)
    if definition is None:
        return _blocked_native_tool_call_response(
            state,
            reason=f"Model requested an unregistered tool: {requested_tool_name}",
            details={"requested_tool_name": requested_tool_name},
        )

    tool_input = dict(getattr(tool_call, "arguments", {}) or {})
    validation_error = _validate_native_tool_input(definition.input_schema, tool_input)
    if validation_error is not None:
        return _blocked_native_tool_call_response(
            state,
            reason=validation_error,
            details={"requested_tool_name": requested_tool_name, "tool_input": tool_input},
        )
    return requested_tool_name, tool_input


def _loop_agent_reflection_retry_input(
    trace_entry: LoopAgentTraceEntry,
    *,
    original_tool_input: dict[str, Any],
    requested_tool_name: str,
    dependencies: AgentGraphDependencies,
) -> dict[str, Any] | None:
    if requested_tool_name != EXTERNAL_WEB_SEARCH_TOOL:
        return None
    reflection = trace_entry.metadata.get("reflection") if isinstance(trace_entry.metadata, dict) else None
    if not isinstance(reflection, dict) or reflection.get("next_action") != "retry":
        return None
    patch = reflection.get("suggested_input_patch")
    if not isinstance(patch, dict) or not patch:
        return None
    retry_input = {**original_tool_input, **patch}
    if retry_input == original_tool_input:
        return None
    definition = dependencies.registry.get(requested_tool_name)
    if definition is None:
        return None
    if _validate_native_tool_input(definition.input_schema, retry_input) is not None:
        return None
    return retry_input


def _reflection_retry_input_from_metadata(
    reflection: dict[str, Any] | None,
    *,
    original_tool_input: dict[str, Any],
    requested_tool_name: str,
    dependencies: AgentGraphDependencies,
) -> dict[str, Any] | None:
    if requested_tool_name != EXTERNAL_WEB_SEARCH_TOOL:
        return None
    if not isinstance(reflection, dict) or reflection.get("next_action") != "retry":
        return None
    patch = reflection.get("suggested_input_patch")
    if not isinstance(patch, dict) or not patch:
        return None
    retry_input = {**original_tool_input, **patch}
    if retry_input == original_tool_input:
        return None
    definition = dependencies.registry.get(requested_tool_name)
    if definition is None:
        return None
    if _validate_native_tool_input(definition.input_schema, retry_input) is not None:
        return None
    return retry_input


def _finalize_native_tool_loop_completion(
    llm_client: Any,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> tuple[Any, LoopAgentStopReason]:
    if tools:
        completion = llm_client.complete(messages=messages, tools=tools, tool_choice="auto")
        if not getattr(completion, "tool_calls", None):
            return completion, LoopAgentStopReason.MODEL_FINAL
        return llm_client.complete(messages=messages), LoopAgentStopReason.BUDGET_EXHAUSTED
    return llm_client.complete(messages=messages), LoopAgentStopReason.MODEL_FINAL


def _finalize_native_tool_loop_after_approval(state: AgentState, *, dependencies: AgentGraphDependencies) -> AgentState:
    if dependencies.llm_client is None or state.current_step == "wait_confirmation" or not state.tool_call_ids:
        return state
    tool_loop = state.context_metadata.get("tool_calling_loop") if isinstance(state.context_metadata, dict) else None
    if not isinstance(tool_loop, dict) or not tool_loop.get("pending_tool_call_id"):
        return state
    final_completion = dependencies.llm_client.complete(messages=state.llm_messages)
    return state.with_updates(
        final_response=final_completion.content,
        response_mode="llm_tool_loop",
        context_metadata=_with_tool_loop_metadata(
            state.context_metadata,
            {
                "executed_tool_name": state.requested_tool_name,
                "tool_call_id": state.tool_call_ids[-1],
                "finalized_after_observation": True,
                "pending_tool_input": None,
            },
        ),
    )


def _build_native_tool_schema_bundle(registry: AgentToolRegistry, allowed_capabilities: list[str]) -> dict[str, Any]:
    tools: list[dict[str, Any]] = []
    alias_to_tool_name: dict[str, str] = {}
    used_aliases: set[str] = set()
    for tool_name in allowed_capabilities:
        definition = registry.get(tool_name)
        if definition is None:
            continue
        alias = _safe_tool_alias(tool_name)
        base_alias = alias
        suffix = 2
        while alias in used_aliases:
            alias = f"{base_alias}_{suffix}"
            suffix += 1
        used_aliases.add(alias)
        alias_to_tool_name[alias] = tool_name
        alias_to_tool_name[tool_name] = tool_name
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": alias,
                    "description": f"{definition.description}\nRegistry tool name: {tool_name}",
                    "parameters": definition.input_schema,
                },
            }
        )
    return {"tools": tools, "alias_to_tool_name": alias_to_tool_name}


def _safe_tool_alias(tool_name: str) -> str:
    alias = re.sub(r"[^A-Za-z0-9_-]", "_", tool_name).strip("_")
    return alias[:64] or "agent_tool"


def _validate_native_tool_input(input_schema: dict[str, Any], tool_input: dict[str, Any]) -> str | None:
    required = input_schema.get("required") if isinstance(input_schema, dict) else None
    if isinstance(required, list):
        missing = [str(name) for name in required if str(name) not in tool_input]
        if missing:
            return f"Model tool call is missing required arguments: {', '.join(missing)}"

    properties = input_schema.get("properties") if isinstance(input_schema, dict) else None
    if input_schema.get("additionalProperties") is False and isinstance(properties, dict):
        extra = sorted(set(tool_input) - set(properties))
        if extra:
            return f"Model tool call included unsupported arguments: {', '.join(extra)}"

    if isinstance(properties, dict):
        for name, schema in properties.items():
            if name not in tool_input or not isinstance(schema, dict):
                continue
            expected_type = schema.get("type")
            if not _matches_json_schema_type(tool_input[name], expected_type):
                return f"Model tool call argument {name} has invalid type."
    return None


def _matches_json_schema_type(value: Any, expected_type: Any) -> bool:
    if expected_type is None:
        return True
    if isinstance(expected_type, list):
        return any(_matches_json_schema_type(value, item) for item in expected_type)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "null":
        return value is None
    return True


def _blocked_native_tool_call_response(state: AgentState, *, reason: str, details: dict[str, Any]) -> AgentState:
    return state.with_updates(
        final_response=f"工具调用已被拦截：{reason}",
        response_mode="tool_call_blocked",
        context_metadata=_with_tool_loop_metadata(
            state.context_metadata,
            {"enabled": True, "blocked": True, "reason": reason, "details": details},
        ),
    )


def _with_tool_loop_metadata(metadata: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    existing = metadata.get("tool_calling_loop") if isinstance(metadata.get("tool_calling_loop"), dict) else {}
    return {**metadata, "tool_calling_loop": {**existing, **updates}}


def _with_loop_agent_metadata(metadata: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    existing = metadata.get("loop_agent") if isinstance(metadata.get("loop_agent"), dict) else {}
    return {**metadata, "loop_agent": {**existing, **updates}}


def _loop_agent_progress_metadata(trace: list[LoopAgentTraceEntry]) -> dict[str, Any]:
    return {
        "enabled": True,
        "control_mode": "runtime_controlled",
        "executed_step_count": len([entry for entry in trace if entry.action == LoopAgentAction.CALL_TOOL]),
        "trace": [entry.to_metadata_dict() for entry in trace],
    }


def _loop_agent_completion_metadata(
    trace: list[LoopAgentTraceEntry],
    *,
    stop_reason: LoopAgentStopReason,
    final_answer: str | None = None,
) -> dict[str, Any]:
    return {
        **_loop_agent_progress_metadata(trace),
        "stop_reason": stop_reason.value,
        "final_answer": final_answer,
        "requires_user_action": stop_reason == LoopAgentStopReason.WAITING_USER,
    }


def _loop_agent_trace_entry(
    iteration: int,
    requested_tool_name: str,
    tool_input: dict[str, Any],
    state: AgentState,
    *,
    dependencies: AgentGraphDependencies,
) -> LoopAgentTraceEntry:
    payload = _latest_tool_result_payload(state, requested_tool_name)
    reflection = _loop_agent_reflection_metadata(
        requested_tool_name,
        tool_input=tool_input,
        payload=payload,
        state=state,
        dependencies=dependencies,
        attempt_index=iteration,
    )
    metadata: dict[str, Any] = {"tool_input_keys": sorted(tool_input.keys())}
    if reflection is not None:
        metadata["reflection"] = reflection
    return LoopAgentTraceEntry(
        iteration=iteration,
        action=LoopAgentAction.CALL_TOOL,
        capability=requested_tool_name,
        decision_reason="Model requested an allowed capability; runtime guard approved or paused the step.",
        observation_status=_loop_agent_observation_status(payload, state),
        observation_summary=_loop_agent_observation_summary(requested_tool_name, payload, state),
        tool_call_id=state.tool_call_ids[-1] if state.tool_call_ids else None,
        metadata=metadata,
    )


def _loop_agent_reflection_metadata(
    requested_tool_name: str,
    *,
    tool_input: dict[str, Any],
    payload: dict[str, Any] | None,
    state: AgentState,
    dependencies: AgentGraphDependencies,
    attempt_index: int,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    definition = dependencies.registry.get(requested_tool_name)
    if definition is None:
        return None
    decision = CapabilityResultEvaluator(llm_client=dependencies.llm_client).evaluate(
        CapabilityResultEvaluationRequest(
            capability=definition,
            tool_input=tool_input,
            result_payload=_reflection_result_payload(payload),
            expected_entities={"company_names": _expected_company_names_from_state(state)},
            task_goal=state.user_message,
            attempt_index=attempt_index,
        )
    )
    return decision.to_metadata_dict() if decision is not None else None


def _reflection_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result_payload = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    return result_payload if isinstance(result_payload, dict) else payload


def _expected_company_names_from_state(state: AgentState) -> list[str]:
    context_pack = state.context_metadata.get("context_pack") if isinstance(state.context_metadata, dict) else None
    intent_frame = context_pack.get("intent_frame") if isinstance(context_pack, dict) else None
    if not isinstance(intent_frame, dict):
        intent_frame = state.context_metadata.get("intent_frame") if isinstance(state.context_metadata, dict) else None
    entities = intent_frame.get("entities") if isinstance(intent_frame, dict) else None
    company_names = entities.get("company_names") if isinstance(entities, dict) else None
    if company_names is None and isinstance(context_pack, dict):
        entities = context_pack.get("entities") if isinstance(context_pack.get("entities"), dict) else None
        company_names = entities.get("company_names") if isinstance(entities, dict) else None
    if isinstance(company_names, list):
        return [str(name).strip() for name in company_names if str(name).strip()]
    return []


def _loop_agent_observation_status(payload: dict[str, Any] | None, state: AgentState) -> str:
    if state.current_step == "wait_confirmation":
        return "waiting_user"
    if isinstance(payload, dict):
        return str(payload.get("status") or "unknown")
    return "unknown"


def _loop_agent_observation_summary(
    requested_tool_name: str,
    payload: dict[str, Any] | None,
    state: AgentState,
) -> str:
    if state.current_step == "wait_confirmation":
        return "Tool step paused because runtime requires user confirmation."
    if not isinstance(payload, dict):
        return "Tool step completed without a structured observation payload."
    tool_result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    result = tool_result.get("result") if isinstance(tool_result.get("result"), dict) else {}
    envelope = tool_result.get("result_envelope") if isinstance(tool_result.get("result_envelope"), dict) else None
    if envelope is None and isinstance(result.get("result_envelope"), dict):
        envelope = result.get("result_envelope")
    if isinstance(envelope, dict) and envelope.get("summary"):
        return str(envelope["summary"])
    if requested_tool_name == EXTERNAL_WEB_SEARCH_TOOL and result.get("answer"):
        return str(result["answer"])
    if requested_tool_name == OFFERIO_COMPANY_JOBS_TOOL:
        return _offerio_sync_summary_response(payload)
    if requested_tool_name == LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL:
        return _company_database_overview_summary_response(payload)
    if requested_tool_name == LOCAL_JOB_SOURCE_OVERVIEW_TOOL:
        return _job_source_overview_summary_response(payload)
    if requested_tool_name == APPLICATION_FIND_APPLY_ENTRY_TOOL:
        return _apply_entry_task_summary_response(payload)
    error = payload.get("error") or tool_result.get("error") or result.get("error")
    if error:
        return str(error)
    return f"Tool step {payload.get('status') or 'completed'}."


def _with_execution_planner_metadata(metadata: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    existing = metadata.get("execution_planner") if isinstance(metadata.get("execution_planner"), dict) else {}
    return {**metadata, "execution_planner": {**existing, **updates}}


def _with_capability_routing_metadata(metadata: dict[str, Any], decision: RouteDecision) -> dict[str, Any]:
    return {**metadata, "capability_routing": decision.to_metadata_dict()}


def _pending_runtime_tool_input(state: AgentState) -> dict[str, Any]:
    planner_input = _pending_execution_planner_tool_input(state)
    if planner_input:
        return planner_input
    return _pending_native_tool_input(state)


def _pending_execution_planner_tool_input(state: AgentState) -> dict[str, Any]:
    planner_metadata = state.context_metadata.get("execution_planner") if isinstance(state.context_metadata, dict) else None
    if not isinstance(planner_metadata, dict):
        return {}
    tool_input = planner_metadata.get("pending_tool_input")
    return dict(tool_input) if isinstance(tool_input, dict) else {}


def _pending_native_tool_input(state: AgentState) -> dict[str, Any]:
    tool_loop = state.context_metadata.get("tool_calling_loop") if isinstance(state.context_metadata, dict) else None
    if not isinstance(tool_loop, dict):
        return {}
    tool_input = tool_loop.get("pending_tool_input")
    return dict(tool_input) if isinstance(tool_input, dict) else {}


def _has_prepared_final_response(state: AgentState) -> bool:
    return bool(state.final_response and state.response_mode != "deterministic_stub")


def _build_context_node(state: AgentState, *, dependencies: AgentGraphDependencies) -> AgentState:
    config = ContextBuildConfig()
    if dependencies.skill_repository is not None:
        dependencies.skill_repository.ensure_builtin_content_source_skills()
    built = MemoryContextBuilder(
        dependencies.conversation_service,
        skill_repository=dependencies.skill_repository,
    ).build(
        state.session_id,
        new_user_message=state.user_message,
        config=config,
    )
    auto_compaction_metadata: dict[str, Any] = {"auto_compacted": False}
    if built.need_compaction:
        try:
            compact_result = dependencies.conversation_service.compact_session(
                state.session_id,
                CompactionConfig(
                    context_window=config.context_window,
                    reserve_tokens=config.reserve_tokens,
                    keep_recent_tokens=config.keep_recent_tokens,
                ),
            )
            built = MemoryContextBuilder(
                dependencies.conversation_service,
                skill_repository=dependencies.skill_repository,
            ).build(
                state.session_id,
                new_user_message=state.user_message,
                config=config,
            )
            auto_compaction_metadata = {
                "auto_compacted": True,
                "auto_compacted_summary_id": compact_result.summary.id,
                "auto_compacted_message_count": compact_result.covered_message_count,
                "auto_compacted_token_estimate_before": compact_result.token_estimate_before,
                "auto_compacted_token_estimate_after": compact_result.token_estimate_after,
            }
        except ValueError as exc:
            auto_compaction_metadata = {
                "auto_compacted": False,
                "auto_compaction_error": str(exc),
            }
    metadata = built.context_metadata
    metadata = {**metadata, **auto_compaction_metadata}
    intent_frame, context_pack = _build_context_pack(
        state.user_message,
        registry=dependencies.registry,
        intent_detector=dependencies.intent_detector,
    )
    metadata = {
        **metadata,
        "intent_frame": intent_frame.model_dump(mode="json"),
        "context_pack": context_pack.to_metadata_dict(),
        "context_engineering": {
            "version": "intent_context_pack_v1",
            "planner_enabled": dependencies.execution_planner is not None,
        },
    }
    llm_messages = _prepend_context_pack_message(built.llm_messages, context_pack)
    return state.with_updates(
        current_step="build_context",
        latest_summary_id=metadata.get("summary_id"),
        loaded_session_history_ids=list(metadata.get("loaded_session_history_ids") or []),
        loaded_memory_ids=list(metadata.get("loaded_memory_ids") or []),
        loaded_skill_ids=list(metadata.get("loaded_skill_ids") or []),
        need_compaction=bool(metadata.get("need_compaction") or False),
        token_estimate=int(metadata.get("token_estimate") or 0),
        llm_messages=llm_messages,
        context_metadata=metadata,
    )


def _build_context_pack(
    user_message: str,
    *,
    registry: AgentToolRegistry,
    intent_detector: Any | None,
) -> tuple[IntentFrame, ContextPack]:
    detector = intent_detector or HybridIntentDetector(llm_client=None)
    try:
        intent_frame = detector.detect(user_message)
    except Exception:
        intent_frame = HybridIntentDetector(llm_client=None).detect(user_message)
    context_pack = ContextPackBuilder(CapabilityCatalog.from_registry(registry)).build(intent_frame)
    return intent_frame, context_pack


def _prepend_context_pack_message(messages: list[dict[str, Any]], context_pack: ContextPack) -> list[dict[str, Any]]:
    payload = context_pack.to_metadata_dict()
    content = (
        "OfferMaster ContextPack for this turn. Use it to decide whether a model-native tool call is appropriate; "
        "the runtime will validate allowed capabilities and tool arguments before execution. Planner is not enabled.\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )
    return [{"role": "system", "content": content, "metadata": {"source": "context_pack"}}, *messages]


def _maybe_tool_node(
    state: AgentState,
    *,
    command: AgentRunCommand,
    dependencies: AgentGraphDependencies,
) -> AgentState:
    skill_permission_policy = _skill_permission_policy_from_state(state)
    guard_result = dependencies.guard.pre_check(
        AgentToolCallContext(
            stage="maybe_tool",
            tool_name=command.requested_tool_name or "",
            source_type=command.source_type,
            tool_call_count=len(state.tool_call_ids),
            user_confirmed=command.user_confirmed,
            agent_run_id=state.agent_run_id,
            session_id=state.session_id,
        ),
        registry=dependencies.registry,
        skill_permission_policy=skill_permission_policy,
    )
    guard_payload = {
        "ok": guard_result.ok,
        "error_code": guard_result.error_code,
        "reason": guard_result.reason,
        "user_message": guard_result.user_message,
        "next_action": guard_result.next_action,
        "retryable": guard_result.retryable,
        "error_details": guard_result.error_details,
        "cost": guard_result.cost,
        "artifacts": guard_result.artifacts,
    }
    if guard_result.next_action == AgentToolNextAction.REQUEST_USER_CONFIRMATION.value:
        return state.with_updates(current_step="wait_confirmation", guard_result=guard_payload)
    if not guard_result.ok:
        return state.with_updates(current_step="maybe_tool", guard_result=guard_payload)

    definition = dependencies.registry.get(command.requested_tool_name or "")
    tool_input = _resolved_tool_input(command, state)
    agent_runtime_executor_id = _agent_runtime_executor_id(command, dependencies=dependencies)
    direct_agent_registered = agent_runtime_executor_id != TOOL_REGISTRY_EXECUTOR_ID and agent_runtime_executor_id in dependencies.agent_executors
    _emit_runtime_tool_event(
        dependencies,
        "tool_started",
        state=state,
        command=command,
        tool_input=tool_input,
        status="running",
        summary=f"开始调用工具：{command.requested_tool_name or 'unknown_tool'}。",
    )
    durable_step_id = _begin_durable_tool_step(
        state,
        command=command,
        dependencies=dependencies,
        tool_input=tool_input,
    )
    if definition is None or dependencies.db_session is None or (definition.handler is None and not direct_agent_registered):
        tool_call = dependencies.automation_service.record_tool_call(
            ToolCallLogCreate(
                workflow_run_id=state.workflow_run_id,
                tool_name=command.requested_tool_name or "",
                tool_group="agent",
                status=ToolCallStatus.FAILED,
                input_payload=tool_input,
                output_payload={"guard_result": guard_payload, "execution": "handler_unavailable"},
                error="Agent tool handler or database session is unavailable.",
            )
        )
        _mark_durable_tool_step_failed(
            durable_step_id,
            state=state,
            command=command,
            tool_input=tool_input,
            dependencies=dependencies,
            tool_call_log_id=tool_call.id,
            output_payload={
                "guard_result": guard_payload,
                "execution": "handler_unavailable",
                "error": tool_call.error,
            },
        )
        _append_tool_pair_messages(
            state,
            dependencies=dependencies,
            tool_call_log_id=tool_call.id,
            tool_name=command.requested_tool_name or "",
            tool_input=tool_input,
            status="failed",
            result=None,
            error=tool_call.error,
        )
        _record_confirmed_skill_approval(state, command=command, dependencies=dependencies)
        _record_skill_runtime_event(
            state,
            dependencies=dependencies,
            event="tool_failed",
            evidence=_tool_runtime_evidence(
                state,
                command=command,
                tool_call_log_id=tool_call.id,
                status="failed",
                guard_payload=guard_payload,
                error=tool_call.error,
            ),
        )
        _emit_runtime_tool_event(
            dependencies,
            "tool_finished",
            state=state,
            command=command,
            tool_input=tool_input,
            tool_call_id=tool_call.id,
            status="failed",
            summary=tool_call.error or f"工具执行失败：{command.requested_tool_name or 'unknown_tool'}。",
        )
        return state.with_updates(
            current_step="maybe_tool",
            guard_result=guard_payload,
            tool_call_ids=[*state.tool_call_ids, tool_call.id],
        )

    try:
        agent_result = _run_agent_tool_through_runtime(
            command,
            state=state,
            dependencies=dependencies,
            tool_input=tool_input,
        )
        result_payload = _jsonable(agent_result.raw_result)
        if not result_payload and agent_result.status == "failed":
            result_payload = {"tool_name": command.requested_tool_name or "", "ok": False, "error": agent_result.summary}
        result_payload = _with_result_envelope(command.requested_tool_name or "", result_payload, state=state)
        tool_ok = agent_result.status != "failed" and _tool_result_ok(result_payload)
        tool_error = None if tool_ok else _tool_result_error(result_payload)
        tool_call = dependencies.automation_service.record_tool_call(
            ToolCallLogCreate(
                workflow_run_id=state.workflow_run_id,
                tool_name=command.requested_tool_name or "",
                tool_group="agent",
                status=ToolCallStatus.SUCCEEDED if tool_ok else ToolCallStatus.FAILED,
                input_payload=tool_input,
                output_payload={
                    "guard_result": guard_payload,
                    "execution": "agent_runtime",
                    "agent_runtime": _agent_runtime_result_metadata(agent_result, executor_id=agent_runtime_executor_id),
                    "result": result_payload,
                },
                error=tool_error,
            )
        )
        _mark_durable_tool_step_completed(
            durable_step_id,
            state=state,
            command=command,
            tool_input=tool_input,
            dependencies=dependencies,
            succeeded=tool_ok,
            tool_call_log_id=tool_call.id,
            external_task_id=_extract_external_task_id(result_payload),
            output_payload={
                "guard_result": guard_payload,
                "execution": "agent_runtime",
                "agent_runtime": _agent_runtime_result_metadata(agent_result, executor_id=agent_runtime_executor_id),
                "result": result_payload,
                "error": tool_error,
            },
        )
        tool_messages = _append_tool_pair_messages(
            state,
            dependencies=dependencies,
            tool_call_log_id=tool_call.id,
            tool_name=command.requested_tool_name or "",
            tool_input=tool_input,
            status="succeeded" if tool_ok else "failed",
            result=result_payload,
            error=tool_error,
        )
        _record_confirmed_skill_approval(state, command=command, dependencies=dependencies)
        _record_skill_runtime_event(
            state,
            dependencies=dependencies,
            event="tool_succeeded",
            evidence=_tool_runtime_evidence(
                state,
                command=command,
                tool_call_log_id=tool_call.id,
                status="succeeded" if tool_ok else "failed",
                guard_payload=guard_payload,
                error=tool_error,
            ),
        )
        _emit_runtime_tool_event(
            dependencies,
            "tool_finished",
            state=state,
            command=command,
            tool_input=tool_input,
            tool_call_id=tool_call.id,
            status="succeeded" if tool_ok else "failed",
            summary=_realtime_tool_finished_summary(
                command.requested_tool_name or "unknown_tool",
                status="succeeded" if tool_ok else "failed",
                result_payload=result_payload,
                error=tool_error,
            ),
        )
    except Exception as exc:  # pragma: no cover - exercised through workflow-level error tests later.
        tool_call = dependencies.automation_service.record_tool_call(
            ToolCallLogCreate(
                workflow_run_id=state.workflow_run_id,
                tool_name=command.requested_tool_name or "",
                tool_group="agent",
                status=ToolCallStatus.FAILED,
                input_payload=tool_input,
                output_payload={
                    "guard_result": guard_payload,
                    "execution": "agent_runtime",
                    "error_type": exc.__class__.__name__,
                },
                error=str(exc),
            )
        )
        _mark_durable_tool_step_failed(
            durable_step_id,
            state=state,
            command=command,
            tool_input=tool_input,
            dependencies=dependencies,
            tool_call_log_id=tool_call.id,
            output_payload={
                "guard_result": guard_payload,
                "execution": "agent_runtime",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
        )
        tool_messages = _append_tool_pair_messages(
            state,
            dependencies=dependencies,
            tool_call_log_id=tool_call.id,
            tool_name=command.requested_tool_name or "",
            tool_input=tool_input,
            status="failed",
            result=None,
            error=str(exc),
        )
        _record_confirmed_skill_approval(state, command=command, dependencies=dependencies)
        _record_skill_runtime_event(
            state,
            dependencies=dependencies,
            event="tool_failed",
            evidence=_tool_runtime_evidence(
                state,
                command=command,
                tool_call_log_id=tool_call.id,
                status="failed",
                guard_payload=guard_payload,
                error=str(exc),
            ),
        )
        _emit_runtime_tool_event(
            dependencies,
            "tool_finished",
            state=state,
            command=command,
            tool_input=tool_input,
            tool_call_id=tool_call.id,
            status="failed",
            summary=str(exc),
        )
    return state.with_updates(
        current_step="maybe_tool",
        guard_result=guard_payload,
        tool_call_ids=[*state.tool_call_ids, tool_call.id],
        llm_messages=[*state.llm_messages, *tool_messages],
    )


def _run_agent_tool_through_runtime(
    command: AgentRunCommand,
    *,
    state: AgentState,
    dependencies: AgentGraphDependencies,
    tool_input: dict[str, Any],
) -> StandardAgentResult:
    executors: dict[str, AbilityAgent] = {
        TOOL_REGISTRY_EXECUTOR_ID: ToolRegistryAgentExecutor(
            dependencies.registry,
            session_provider=lambda _context: dependencies.db_session,
        ),
        **dependencies.agent_executors,
    }
    runtime = AgentRuntime(
        registry=create_default_agent_capability_registry(
            tool_registry=dependencies.registry,
            executor_id_by_capability=dependencies.capability_executor_ids,
        ),
        executors=executors,
    )
    return runtime.call(
        AgentTask(
            capability_id=command.requested_tool_name or "",
            goal=state.user_message,
            input_payload=tool_input,
        ),
        AgentRuntimeContext(
            session_id=state.session_id,
            run_id=state.workflow_run_id,
            task_id=f"{state.workflow_run_id}:tool-{len(state.tool_call_ids) + 1}",
            permission_scope={"source_type": command.source_type, "user_confirmed": command.user_confirmed},
            metadata={"agent_run_id": state.agent_run_id},
        ),
    )


def _agent_runtime_executor_id(command: AgentRunCommand, *, dependencies: AgentGraphDependencies) -> str:
    capability = command.requested_tool_name or ""
    return dependencies.capability_executor_ids.get(capability, TOOL_REGISTRY_EXECUTOR_ID)


def _emit_runtime_tool_event(
    dependencies: AgentGraphDependencies,
    event_type: str,
    *,
    state: AgentState,
    command: AgentRunCommand,
    tool_input: dict[str, Any],
    status: str,
    summary: str,
    tool_call_id: str | None = None,
    reflection: dict[str, Any] | None = None,
    suggested_input_patch: dict[str, Any] | None = None,
) -> None:
    if dependencies.event_sink is None:
        return
    tool_name = command.requested_tool_name or "unknown_tool"
    payload: dict[str, Any] = {
        "event_type": event_type,
        "event_label": _runtime_tool_event_label(event_type),
        "session_id": state.session_id,
        "workflow_run_id": state.workflow_run_id,
        "agent_run_id": state.agent_run_id,
        "step_index": len(state.tool_call_ids) + 1,
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "status": status,
        "summary": summary,
        "tool_input_keys": sorted(str(key) for key in tool_input.keys()),
    }
    if reflection is not None:
        payload["reflection"] = dict(reflection)
    if suggested_input_patch is not None:
        payload["suggested_input_patch"] = dict(suggested_input_patch)
    dependencies.event_sink(payload)


def _runtime_tool_event_label(event_type: str) -> str:
    return {
        "tool_started": "工具开始",
        "tool_finished": "工具完成",
        "tool_reflection_retry": "准备重试",
    }.get(event_type, event_type)


def _realtime_tool_finished_summary(
    tool_name: str,
    *,
    status: str,
    result_payload: Any,
    error: str | None,
) -> str:
    if error:
        return error
    if tool_name == EXTERNAL_WEB_SEARCH_TOOL:
        return f"工具执行完成：{tool_name}，状态：{status}。"
    if isinstance(result_payload, dict):
        envelope = result_payload.get("result_envelope") if isinstance(result_payload.get("result_envelope"), dict) else None
        if envelope is not None and envelope.get("summary"):
            return str(envelope["summary"])
    return f"工具执行完成：{tool_name}，状态：{status}。"


def _emit_loop_reflection_retry_event(
    dependencies: AgentGraphDependencies,
    *,
    state: AgentState,
    requested_tool_name: str,
    retry_input: dict[str, Any],
    trace_entry: LoopAgentTraceEntry,
) -> None:
    reflection = trace_entry.metadata.get("reflection") if isinstance(trace_entry.metadata, dict) else None
    if not isinstance(reflection, dict):
        reflection = {}
    summary = str(reflection.get("reason") or "工具结果不够好，准备修改输入后重试。")
    _emit_runtime_tool_event(
        dependencies,
        "tool_reflection_retry",
        state=state,
        command=AgentRunCommand(
            session_id=state.session_id,
            user_message=state.user_message,
            requested_tool_name=requested_tool_name,
            source_type="agent_chat",
            tool_input=retry_input,
        ),
        tool_input=retry_input,
        tool_call_id=trace_entry.tool_call_id,
        status="retry",
        summary=summary,
        reflection=reflection,
        suggested_input_patch=retry_input,
    )


def _agent_runtime_result_metadata(result: StandardAgentResult, *, executor_id: str) -> dict[str, Any]:
    return {
        "executor_id": executor_id,
        "status": result.status,
        "summary": result.summary,
        "requires_user_action": result.requires_user_action,
    }


def _begin_durable_tool_step(
    state: AgentState,
    *,
    command: AgentRunCommand,
    dependencies: AgentGraphDependencies,
    tool_input: dict[str, Any],
) -> str | None:
    if dependencies.durable_state_service is None:
        return None
    sequence_index = len(state.tool_call_ids) + 1
    return f"{state.workflow_run_id}:tool-{sequence_index}"


def _mark_durable_tool_step_completed(
    step_id: str | None,
    *,
    state: AgentState,
    command: AgentRunCommand,
    tool_input: dict[str, Any],
    dependencies: AgentGraphDependencies,
    succeeded: bool,
    tool_call_log_id: str,
    external_task_id: str | None,
    output_payload: dict[str, Any],
) -> None:
    service = dependencies.durable_state_service
    if service is None or step_id is None:
        return
    try:
        _ensure_durable_tool_step(
            step_id,
            state=state,
            command=command,
            tool_input=tool_input,
            dependencies=dependencies,
        )
        if succeeded:
            service.mark_step_succeeded(
                step_id,
                tool_call_log_id=tool_call_log_id,
                external_task_id=external_task_id,
                output_payload=output_payload,
            )
            result_payload = output_payload.get("result")
            _record_durable_memory_snapshots(
                service,
                task_id=state.workflow_run_id,
                step_id=step_id,
                tool_name=command.requested_tool_name or "",
                result_payload=result_payload,
            )
            _record_durable_artifacts(
                service,
                task_id=state.workflow_run_id,
                step_id=step_id,
                result_payload=result_payload,
            )
            if external_task_id:
                service.sync_external_agent_artifacts(
                    task_id=state.workflow_run_id,
                    step_id=step_id,
                    external_task_id=external_task_id,
                )
            return
        service.mark_step_failed(
            step_id,
            output_payload={**output_payload, "tool_call_log_id": tool_call_log_id},
        )
    except Exception:
        return


def _mark_durable_tool_step_failed(
    step_id: str | None,
    *,
    state: AgentState,
    command: AgentRunCommand,
    tool_input: dict[str, Any],
    dependencies: AgentGraphDependencies,
    tool_call_log_id: str,
    output_payload: dict[str, Any],
) -> None:
    service = dependencies.durable_state_service
    if service is None or step_id is None:
        return
    try:
        _ensure_durable_tool_step(
            step_id,
            state=state,
            command=command,
            tool_input=tool_input,
            dependencies=dependencies,
        )
        service.mark_step_failed(
            step_id,
            output_payload={**output_payload, "tool_call_log_id": tool_call_log_id},
        )
    except Exception:
        return


def _ensure_durable_tool_step(
    step_id: str,
    *,
    state: AgentState,
    command: AgentRunCommand,
    tool_input: dict[str, Any],
    dependencies: AgentGraphDependencies,
) -> None:
    service = dependencies.durable_state_service
    if service is None:
        return
    task_id = state.workflow_run_id
    tool_name = command.requested_tool_name or "unknown_tool"
    try:
        task = service.get_task(task_id)
        if getattr(task, "capability", None) == "agent.context_builder":
            task.task_type = "agent_tool_execution"
            task.capability = tool_name
            task.owner_executor = command.source_type
            task.user_goal = state.user_message
            task.input_payload = {
                **dict(task.input_payload or {}),
                "agent_run_id": state.agent_run_id,
                "requested_tool_name": tool_name,
                "source_type": command.source_type,
            }
            service.repository.update_task(task)
    except DurableStateNotFoundError:
        service.create_task(
            task_id=task_id,
            root_workflow_run_id=state.workflow_run_id,
            conversation_session_id=state.session_id,
            task_type="agent_tool_execution",
            capability=tool_name,
            owner_executor=command.source_type,
            user_goal=state.user_message,
            input_payload={
                "agent_run_id": state.agent_run_id,
                "requested_tool_name": tool_name,
                "source_type": command.source_type,
            },
        )
    try:
        service.get_step(step_id)
    except DurableStateNotFoundError:
        service.add_step(
            task_id=task_id,
            step_id=step_id,
            sequence_index=len(state.tool_call_ids) + 1,
            step_type="tool_call",
            executor_type="tool_registry",
            executor_name="agent_tool_registry",
            capability=tool_name,
            input_payload={
                "agent_run_id": state.agent_run_id,
                "requested_tool_name": tool_name,
                "source_type": command.source_type,
                "tool_input": tool_input,
            },
        )
    service.mark_step_running(step_id)


def _record_durable_context_snapshots(state: AgentState, *, dependencies: AgentGraphDependencies) -> None:
    service = dependencies.durable_state_service
    if service is None:
        return
    refs = _context_snapshot_refs(state)
    if not refs:
        return
    step_id = f"{state.workflow_run_id}:context"
    try:
        _ensure_durable_context_step(step_id, state=state, dependencies=dependencies)
        for ref in refs:
            service.record_memory_snapshot(
                snapshot_id=f"memory-snapshot-{uuid4()}",
                task_id=state.workflow_run_id,
                step_id=step_id,
                memory_id=ref["memory_id"],
                source_type=ref["source_type"],
                usage_reason=ref["usage_reason"],
                visibility_scope=ref["visibility_scope"],
                passed_to_executor=False,
                memory_payload=ref["memory_payload"],
            )
    except Exception:
        return


def _ensure_durable_context_step(
    step_id: str,
    *,
    state: AgentState,
    dependencies: AgentGraphDependencies,
) -> None:
    service = dependencies.durable_state_service
    if service is None:
        return
    try:
        service.get_task(state.workflow_run_id)
    except DurableStateNotFoundError:
        service.create_task(
            task_id=state.workflow_run_id,
            root_workflow_run_id=state.workflow_run_id,
            conversation_session_id=state.session_id,
            task_type="agent_context_build",
            capability="agent.context_builder",
            owner_executor="offermaster_runtime",
            user_goal=state.user_message,
            input_payload={
                "agent_run_id": state.agent_run_id,
                "context_metadata_keys": sorted(state.context_metadata.keys()),
            },
        )
    try:
        service.get_step(step_id)
        return
    except DurableStateNotFoundError:
        service.add_step(
            task_id=state.workflow_run_id,
            step_id=step_id,
            sequence_index=0,
            step_type="context_build",
            executor_type="runtime",
            executor_name="offermaster_context_builder",
            capability="agent.context_builder",
            input_payload={
                "loaded_session_history_ids": list(state.loaded_session_history_ids),
                "loaded_skill_ids": list(state.loaded_skill_ids),
                "latest_summary_id": state.latest_summary_id,
            },
        )
        service.mark_step_succeeded(
            step_id,
            output_payload={
                "token_estimate": state.token_estimate,
                "need_compaction": state.need_compaction,
            },
        )


def _context_snapshot_refs(state: AgentState) -> list[dict[str, Any]]:
    from app.agent_runtime.durable_state.schemas import AgentMemoryVisibilityScope

    refs: list[dict[str, Any]] = []
    for skill_id in state.loaded_skill_ids:
        refs.append(
            {
                "memory_id": str(skill_id),
                "source_type": "agent_skill",
                "usage_reason": "ContextBuilder loaded skill for this run",
                "visibility_scope": AgentMemoryVisibilityScope.MAIN_AGENT_ONLY,
                "memory_payload": {"skill_id": str(skill_id)},
            }
        )
    for message_id in state.loaded_session_history_ids:
        refs.append(
            {
                "memory_id": str(message_id),
                "source_type": "session_history",
                "usage_reason": "ContextBuilder loaded session history for this run",
                "visibility_scope": AgentMemoryVisibilityScope.RUNTIME_ONLY,
                "memory_payload": {"message_id": str(message_id)},
            }
        )
    if state.latest_summary_id:
        refs.append(
            {
                "memory_id": str(state.latest_summary_id),
                "source_type": "context_summary",
                "usage_reason": "ContextBuilder loaded compacted session summary for this run",
                "visibility_scope": AgentMemoryVisibilityScope.RUNTIME_ONLY,
                "memory_payload": {"summary_id": str(state.latest_summary_id)},
            }
        )
    return refs


def _record_durable_memory_snapshots(
    service: Any,
    *,
    task_id: str,
    step_id: str,
    tool_name: str,
    result_payload: Any,
) -> None:
    if not isinstance(result_payload, dict):
        return
    if tool_name == "memory_search":
        query = str(result_payload.get("query") or "")
        for item in result_payload.get("items") or []:
            if not isinstance(item, dict) or not item.get("memory_id"):
                continue
            service.record_memory_snapshot(
                snapshot_id=f"memory-snapshot-{uuid4()}",
                task_id=task_id,
                step_id=step_id,
                memory_id=str(item["memory_id"]),
                source_type=str(item.get("source_type") or "unknown"),
                usage_reason=f"memory_search matched query: {query}",
                passed_to_executor=False,
                memory_payload={"excerpt": item.get("excerpt"), "score": item.get("score")},
            )
    if tool_name == "memory_get" and result_payload.get("found") and result_payload.get("memory_id"):
        service.record_memory_snapshot(
            snapshot_id=f"memory-snapshot-{uuid4()}",
            task_id=task_id,
            step_id=step_id,
            memory_id=str(result_payload["memory_id"]),
            source_type=str(result_payload.get("source_type") or "unknown"),
            usage_reason="memory_get loaded exact memory",
            passed_to_executor=False,
            memory_payload={
                "excerpt": result_payload.get("excerpt"),
                "metadata": result_payload.get("metadata"),
            },
        )


def _record_durable_artifacts(
    service: Any,
    *,
    task_id: str,
    step_id: str,
    result_payload: Any,
) -> None:
    if not isinstance(result_payload, dict):
        return
    envelope = result_payload.get("result_envelope")
    if not isinstance(envelope, dict):
        nested = result_payload.get("result")
        envelope = nested.get("result_envelope") if isinstance(nested, dict) else None
    if isinstance(envelope, dict):
        service.record_artifacts_from_result_envelope(
            task_id=task_id,
            step_id=step_id,
            result_envelope=envelope,
        )


def _extract_external_task_id(result_payload: Any) -> str | None:
    if not isinstance(result_payload, dict):
        return None
    for key in ("external_task_id", "task_id"):
        value = result_payload.get(key)
        if value:
            return str(value)
    result = result_payload.get("result")
    if isinstance(result, dict):
        for key in ("external_task_id", "task_id"):
            value = result.get(key)
            if value:
                return str(value)
    return None


def _skill_permission_policy_from_state(state: AgentState) -> AgentToolPermissionPolicy | None:
    snapshot = state.context_metadata.get("skill_tool_permission_policy")
    if not isinstance(snapshot, dict):
        return None
    if not any(snapshot.get(key) for key in ("skill_ids", "allowed_tools", "ask_tools", "disallowed_tools")):
        return None
    return AgentToolPermissionPolicy.from_metadata_snapshot(snapshot)


def _record_skill_runtime_event(
    state: AgentState,
    *,
    dependencies: AgentGraphDependencies,
    event: str,
    evidence: dict[str, Any] | None = None,
) -> None:
    if dependencies.skill_repository is None:
        return
    for skill_id in _runtime_event_skill_ids(state):
        dependencies.skill_repository.record_runtime_event(skill_id, event=event, evidence=evidence)


def _record_confirmed_skill_approval(
    state: AgentState,
    *,
    command: AgentRunCommand,
    dependencies: AgentGraphDependencies,
) -> None:
    if not command.user_confirmed or not state.approval_request_id:
        return
    approval = dependencies.db_session.get(ApprovalRequest, state.approval_request_id) if dependencies.db_session is not None else None
    _record_skill_runtime_event(
        state,
        dependencies=dependencies,
        event="approval_approved",
        evidence={
            "approval_request_id": state.approval_request_id,
            "workflow_run_id": state.workflow_run_id,
            "agent_run_id": state.agent_run_id,
            "tool_name": command.requested_tool_name,
            "decision_reason": approval.decision if approval is not None else None,
        },
    )


def _runtime_event_skill_ids(state: AgentState) -> list[str]:
    skill_ids: list[str] = []
    snapshot = state.context_metadata.get("skill_tool_permission_policy")
    if isinstance(snapshot, dict):
        skill_ids.extend(str(skill_id).strip() for skill_id in snapshot.get("skill_ids") or [] if str(skill_id).strip())
    skill_ids.extend(str(skill_id).strip() for skill_id in state.loaded_skill_ids if str(skill_id).strip())
    guard_result = state.guard_result if isinstance(state.guard_result, dict) else {}
    error_details = guard_result.get("error_details") if isinstance(guard_result.get("error_details"), dict) else {}
    skill_ids.extend(str(skill_id).strip() for skill_id in error_details.get("skill_ids") or [] if str(skill_id).strip())
    return list(dict.fromkeys(skill_ids))


def _tool_runtime_evidence(
    state: AgentState,
    *,
    command: AgentRunCommand,
    tool_call_log_id: str,
    status: str,
    guard_payload: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    error_details = guard_payload.get("error_details") if isinstance(guard_payload.get("error_details"), dict) else {}
    return {
        "workflow_run_id": state.workflow_run_id,
        "agent_run_id": state.agent_run_id,
        "approval_request_id": state.approval_request_id,
        "tool_call_log_id": tool_call_log_id,
        "tool_name": command.requested_tool_name,
        "source_type": command.source_type,
        "status": status,
        "guard_error_code": guard_payload.get("error_code"),
        "permission_decision": error_details.get("permission_decision"),
        "error": error,
    }


def _resolved_tool_input(command: AgentRunCommand, state: AgentState) -> dict[str, Any]:
    if command.tool_input:
        return dict(command.tool_input)
    if command.requested_tool_name == "weixin-articles-mcp.read_article":
        url = _extract_weixin_article_url(state.user_message)
        return {"url": url} if url else {}
    if command.requested_tool_name == "xiaohongshu-mcp.search_feeds":
        return {"keyword": _xiaohongshu_keyword(state.user_message)}
    if command.requested_tool_name == "xiaohongshu-mcp.get_feed_detail":
        detail_input = _extract_xiaohongshu_detail_input(state.user_message)
        return detail_input or {}
    if command.requested_tool_name == EXTERNAL_WEB_SEARCH_TOOL:
        return {"query": state.user_message, "max_results": 5}
    if command.requested_tool_name in {"sessions_search", "memory_search"}:
        return {"query": state.user_message, "limit": 10}
    if command.requested_tool_name == "sessions_history":
        return {"session_key": state.session_id, "window_before": 5, "window_after": 5}
    if command.requested_tool_name == OFFERIO_COMPANY_JOBS_TOOL:
        return {"limit": 1000}
    if command.requested_tool_name == LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL:
        return {"sample_limit": requested_sample_limit_from_text(state.user_message)}
    if command.requested_tool_name == LOCAL_JOB_SOURCE_OVERVIEW_TOOL:
        return {"sample_limit": requested_sample_limit_from_text(state.user_message), "include_external_job_board": True}
    if command.requested_tool_name == APPLICATION_FIND_APPLY_ENTRY_TOOL:
        job_id = _extract_application_job_id(state.user_message)
        return {"job_id": job_id} if job_id else {}
    return {}


def _auto_select_tool_command(command: AgentRunCommand, *, state: AgentState, registry: AgentToolRegistry) -> AgentRunCommand:
    if command.requested_tool_name:
        return command

    weixin_url = _extract_weixin_article_url(command.user_message)
    if weixin_url and registry.get("weixin-articles-mcp.read_article") is not None:
        return replace(
            command,
            requested_tool_name="weixin-articles-mcp.read_article",
            source_type="wechat_article",
            tool_input={"url": weixin_url},
        )

    xiaohongshu_detail = _extract_xiaohongshu_detail_input(command.user_message)
    if xiaohongshu_detail and registry.get("xiaohongshu-mcp.get_feed_detail") is not None:
        return replace(
            command,
            requested_tool_name="xiaohongshu-mcp.get_feed_detail",
            source_type="xiaohongshu_note",
            tool_input=xiaohongshu_detail,
        )

    return command


_WEIXIN_ARTICLE_URL_RE = re.compile(r"https?://mp\.weixin\.qq\.com/[^\s)）>\]]+", re.IGNORECASE)
_XIAOHONGSHU_FEED_ID_RE = re.compile(r"(?:feed_id|note_id|item_id)\s*[=:\uff1a]\s*([a-zA-Z0-9_-]+)")
_XIAOHONGSHU_XSEC_RE = re.compile(r"xsec_token\s*[=:\uff1a]\s*([^\s&\uff0c,]+)")
_APPLICATION_JOB_ID_RE = re.compile(r"(?:job_id|jobId|lead_id|leadId|\u5c97\u4f4did|\u5c97\u4f4dID)\s*[=:\uff1a]\s*([A-Za-z0-9][A-Za-z0-9_.:-]{0,127})")


def _extract_weixin_article_url(text: str) -> str | None:
    match = _WEIXIN_ARTICLE_URL_RE.search(text)
    if match is None:
        return None
    return match.group(0).rstrip("。.,，、;；")


def _extract_xiaohongshu_detail_input(text: str) -> dict[str, Any] | None:
    feed_id = _XIAOHONGSHU_FEED_ID_RE.search(text)
    xsec_token = _XIAOHONGSHU_XSEC_RE.search(text)
    if feed_id is None or xsec_token is None:
        return None
    return {"feed_id": feed_id.group(1), "xsec_token": xsec_token.group(1)}


def _xiaohongshu_keyword(text: str) -> str:
    keyword = re.sub(r"https?://\S+", " ", text).strip()
    keyword = re.sub(r"\s+", " ", keyword)
    return keyword or text.strip()


def _extract_application_job_id(text: str) -> str | None:
    match = _APPLICATION_JOB_ID_RE.search(text)
    if match is None:
        return None
    return match.group(1).rstrip("\u3002.,\uff0c\u3001;\uff1b")


def _append_tool_pair_messages(
    state: AgentState,
    *,
    dependencies: AgentGraphDependencies,
    tool_call_log_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    status: str,
    result: Any,
    error: str | None,
) -> list[dict[str, Any]]:
    call_message = dependencies.conversation_service.append_message(
        state.session_id,
        AgentMessageCreate(
            role=AgentMessageRole.TOOL_CALL,
            content_text=f"Tool call: {tool_name}",
            content_json={"tool_name": tool_name, "input": tool_input},
            agent_run_id=state.agent_run_id,
            workflow_run_id=state.workflow_run_id,
            tool_call_log_id=tool_call_log_id,
            token_estimate=0,
        ),
    )
    result_message = dependencies.conversation_service.append_message(
        state.session_id,
        AgentMessageCreate(
            role=AgentMessageRole.TOOL_RESULT,
            content_text=f"Tool result: {tool_name} {status}",
            content_json={"tool_name": tool_name, "status": status, "result": result, "error": error},
            agent_run_id=state.agent_run_id,
            workflow_run_id=state.workflow_run_id,
            tool_call_log_id=tool_call_log_id,
            parent_message_id=call_message.id,
            token_estimate=0,
        ),
    )
    return [_message_to_llm_context(call_message), _message_to_llm_context(result_message)]


def _message_to_llm_context(message) -> dict[str, Any]:
    content = message.content_text or ""
    metadata: dict[str, Any] = {"message_id": message.id, "source": "tool_transcript"}
    content_json = _jsonable(getattr(message, "content_json", None))
    if isinstance(content_json, dict):
        metadata.update(
            {
                "tool_name": content_json.get("tool_name"),
                "tool_status": content_json.get("status"),
                "content_json": content_json,
            }
        )
        content = f"{content}\n{json.dumps(content_json, ensure_ascii=False, separators=(',', ':'))}"
    return {
        "role": "assistant",
        "content": content,
        "metadata": metadata,
    }


def _tool_result_ok(result_payload: Any) -> bool:
    if isinstance(result_payload, dict) and "ok" in result_payload:
        return bool(result_payload.get("ok"))
    return True


def _tool_result_error(result_payload: Any) -> str:
    if not isinstance(result_payload, dict):
        return "Agent tool returned a failed result."
    error = result_payload.get("error")
    if error:
        return str(error)
    nested_result = result_payload.get("result")
    if isinstance(nested_result, dict):
        nested_error = nested_result.get("error") or nested_result.get("message")
        if nested_error:
            return str(nested_error)
    return "Agent tool returned ok=false without a detailed error."


def _with_result_envelope(tool_name: str, result_payload: Any, *, state: AgentState) -> Any:
    if not isinstance(result_payload, dict) or "result_envelope" in result_payload:
        return result_payload
    status = "succeeded" if _tool_result_ok(result_payload) else "failed"
    context_pack = state.context_metadata.get("context_pack") if isinstance(state.context_metadata, dict) else None
    risk_level = str(context_pack.get("risk_level") or "low") if isinstance(context_pack, dict) else "low"
    envelope = build_result_envelope(
        capability=tool_name,
        status=status,
        result_payload=result_payload,
        risk_level=risk_level,
    )
    if envelope is None:
        return result_payload
    return {**result_payload, "result_envelope": envelope.to_dict()}


def _generate_final_response(state: AgentState, *, dependencies: AgentGraphDependencies) -> tuple[str, str]:
    unreliable_search_response = _unreliable_external_web_search_response(state)
    if unreliable_search_response is not None:
        return unreliable_search_response, "tool_result_summary_unreliable"
    if _has_prepared_final_response(state):
        return state.final_response, state.response_mode
    synthesis_messages = external_web_search_synthesis_messages(state)
    if synthesis_messages is not None and dependencies.llm_client is not None:
        try:
            completion = dependencies.llm_client.complete(messages=synthesis_messages)
            return completion.content, "llm_tool_result_summary"
        except Exception:
            pass
    tool_response = tool_result_summary_response(state)
    if tool_response is not None:
        return tool_response
    if dependencies.llm_client is None:
        return "Agent runtime completed deterministic workflow skeleton.", "deterministic_stub"
    completion = dependencies.llm_client.complete(messages=state.llm_messages)
    return completion.content, "llm"


def tool_result_summary_response(state: AgentState) -> tuple[str, str] | None:
    if state.requested_tool_name == LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL:
        payload = _latest_tool_result_payload(state, LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL)
        if payload is None:
            return None
        return _company_database_overview_summary_response(payload), "tool_result_summary"
    if state.requested_tool_name == LOCAL_JOB_SOURCE_OVERVIEW_TOOL:
        payload = _latest_tool_result_payload(state, LOCAL_JOB_SOURCE_OVERVIEW_TOOL)
        if payload is None:
            return None
        return _job_source_overview_summary_response(payload), "tool_result_summary"
    if state.requested_tool_name == OFFERIO_COMPANY_JOBS_TOOL:
        payload = _latest_tool_result_payload(state, OFFERIO_COMPANY_JOBS_TOOL)
        if payload is None:
            return None
        return _offerio_sync_summary_response(payload), "tool_result_summary"
    if state.requested_tool_name == APPLICATION_FIND_APPLY_ENTRY_TOOL:
        payload = _latest_tool_result_payload(state, APPLICATION_FIND_APPLY_ENTRY_TOOL)
        if payload is None:
            return None
        return _apply_entry_task_summary_response(payload), "tool_result_summary"
    if state.requested_tool_name == EXTERNAL_WEB_SEARCH_TOOL:
        unreliable_search_response = _unreliable_external_web_search_response(state)
        if unreliable_search_response is not None:
            return unreliable_search_response, "tool_result_summary_unreliable"
        payload = _latest_tool_result_payload(state, EXTERNAL_WEB_SEARCH_TOOL)
        if payload is None:
            return None
        return _external_web_search_summary_response(payload), "tool_result_summary"
    return None


def external_web_search_synthesis_messages(state: AgentState) -> list[dict[str, Any]] | None:
    if state.requested_tool_name != EXTERNAL_WEB_SEARCH_TOOL:
        return None
    if _unreliable_external_web_search_response(state) is not None:
        return None
    payload = _latest_tool_result_payload(state, EXTERNAL_WEB_SEARCH_TOOL)
    if payload is None or not _external_web_search_result_ok(payload):
        return None

    raw_result = _external_web_search_result_payload(payload)
    answer = str(raw_result.get("answer") or "").strip()
    sources = raw_result.get("sources") if isinstance(raw_result.get("sources"), list) else []
    if not answer and not sources:
        return None

    evidence = {
        "query": raw_result.get("query"),
        "answer": answer,
        "sources": sources,
        "executor_name": raw_result.get("executor_name"),
    }
    instruction = (
        "你是 OfferMaster 的主 Agent，正在根据 external.web_search 的原始搜索结果回答用户。"
        "遵循 retrieval-augmented generation 的证据使用方式：先理解用户任务，再基于检索证据回答。"
        "不要把搜索结果原样照抄给用户，必须先筛选、去噪、归纳。"
        "如果用户问的是校园招聘、秋招或岗位信息，优先保留官方招聘站、公司官网招聘页、可信高校就业网；"
        "内部忽略百科、泛公司介绍、体育球队、NBA、同名无关实体、广告页和没有招聘信息的结果。"
        "不要向用户展示无关结果，不要解释过滤过程，不要列出被忽略的来源；"
        "不要提及无关结果的标题、类型、数量或分类，例如不要写‘几条百度百科’、‘几条我的世界’、‘这些结果说明’。"
        "不能因为检索结果全是无关内容，就推断目标公司尚未发布招聘；只能说本次检索未找到可靠公开证据或明确入口。"
        "只有用户明确追问为什么排除某些结果时，才简要说明。"
        "只基于搜索结果作答，不要编造没有证据的开放时间、岗位数量或投递要求。"
        "若没有找到可靠招聘证据，直接说明没有找到明确入口，并给出下一步建议。"
        "用中文回答，结构简洁，只保留对用户有用的关键链接。"
    )
    evidence_message = (
        f"用户原始问题：{state.user_message}\n\n"
        "external.web_search 原始结果 JSON：\n"
        f"{json.dumps(evidence, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "请基于这些结果给出最终回答。"
    )
    return [
        {"role": "system", "content": instruction, "metadata": {"source": "external_web_search_synthesis"}},
        *state.llm_messages,
        {"role": "user", "content": evidence_message, "metadata": {"source": "external_web_search_synthesis"}},
    ]


def _latest_tool_result_payload(state: AgentState, tool_name: str) -> dict[str, Any] | None:
    for message in reversed(state.llm_messages):
        metadata = message.get("metadata") if isinstance(message, dict) else None
        content_json = metadata.get("content_json") if isinstance(metadata, dict) else None
        if isinstance(content_json, dict) and content_json.get("tool_name") == tool_name and "status" in content_json:
            return content_json

        content = str(message.get("content") or "") if isinstance(message, dict) else ""
        if "\n" not in content:
            continue
        _, json_text = content.split("\n", 1)
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("tool_name") == tool_name and "status" in parsed:
            return parsed
    return None


def _external_web_search_result_ok(payload: dict[str, Any]) -> bool:
    tool_result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    return payload.get("status") == "succeeded" and _tool_result_ok(tool_result)


def _external_web_search_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    tool_result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    return tool_result.get("result") if isinstance(tool_result.get("result"), dict) else {}


def _unreliable_external_web_search_response(state: AgentState) -> str | None:
    if state.requested_tool_name != EXTERNAL_WEB_SEARCH_TOOL:
        return None
    reflection = _latest_external_web_search_reflection(state)
    if reflection is None:
        return None
    quality = str(reflection.get("quality") or "").lower()
    next_action = str(reflection.get("next_action") or "").lower()
    if quality == "good" or next_action != "retry":
        return None
    attempted_count = len(_external_web_search_attempted_queries_from_loop(state))
    count_text = f"，并尝试换关键词重试了 {attempted_count} 次" if attempted_count > 1 else ""
    return (
        f"我已经调用联网搜索{count_text}，但这轮搜索结果仍然和你的问题不匹配，"
        "没有找到可靠公开证据可以回答。为了避免误导，我不直接编造具体赛程或结论。"
        "你可以优先查看官方或权威体育来源，例如 Al Nassr 官网、ESPN、Flashscore、SofaScore；"
        "如果你指定更明确的日期范围或赛事类型，我可以继续按这个范围重新查。"
    )


def _latest_external_web_search_reflection(state: AgentState) -> dict[str, Any] | None:
    for trace_entry in reversed(_external_web_search_loop_trace_entries(state)):
        reflection = _reflection_from_trace_entry(trace_entry)
        if reflection is not None:
            return reflection
    return None


def _external_web_search_attempted_queries_from_loop(state: AgentState) -> list[str]:
    queries: list[str] = []
    for trace_entry in _external_web_search_loop_trace_entries(state):
        metadata = trace_entry.get("metadata") if isinstance(trace_entry, dict) else None
        if not isinstance(metadata, dict):
            continue
        observation = metadata.get("observation") if isinstance(metadata.get("observation"), dict) else None
        observation_metadata = observation.get("metadata") if isinstance(observation, dict) else None
        tool_input = observation_metadata.get("tool_input") if isinstance(observation_metadata, dict) else None
        query = str(tool_input.get("query") or "").strip() if isinstance(tool_input, dict) else ""
        if query:
            queries.append(query)
    return queries


def _external_web_search_loop_trace_entries(state: AgentState) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not isinstance(state.context_metadata, dict):
        return entries
    for metadata_key in ("tool_choice_loop", "loop_agent"):
        loop_metadata = state.context_metadata.get(metadata_key)
        if not isinstance(loop_metadata, dict):
            continue
        trace = loop_metadata.get("trace")
        if not isinstance(trace, list):
            continue
        for entry in trace:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("capability") or "") == EXTERNAL_WEB_SEARCH_TOOL:
                entries.append(entry)
    return entries


def _reflection_from_trace_entry(trace_entry: dict[str, Any]) -> dict[str, Any] | None:
    metadata = trace_entry.get("metadata") if isinstance(trace_entry.get("metadata"), dict) else {}
    direct_reflection = metadata.get("reflection")
    if isinstance(direct_reflection, dict):
        return direct_reflection
    observation = metadata.get("observation") if isinstance(metadata.get("observation"), dict) else None
    if not isinstance(observation, dict):
        return None
    observation_metadata = observation.get("metadata") if isinstance(observation.get("metadata"), dict) else {}
    observation_reflection = observation_metadata.get("reflection")
    if isinstance(observation_reflection, dict):
        return observation_reflection
    suggested = observation.get("suggested_next_decision")
    if isinstance(suggested, dict):
        suggested_metadata = suggested.get("metadata") if isinstance(suggested.get("metadata"), dict) else {}
        suggested_reflection = suggested_metadata.get("reflection")
        if isinstance(suggested_reflection, dict):
            return suggested_reflection
    return None


def _offerio_sync_summary_response(payload: dict[str, Any]) -> str:
    tool_result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    result = tool_result.get("result") if isinstance(tool_result.get("result"), dict) else {}
    source_name = str(result.get("source_name") or "OfferIO 公司聚合岗位库")
    status = str(result.get("status") or payload.get("status") or "unknown")
    error = payload.get("error") or tool_result.get("error") or result.get("error")
    ok = payload.get("status") == "succeeded" and _tool_result_ok(tool_result)
    if not ok:
        return f"{source_name} 同步失败：{error or status}。"

    fetched_count = _safe_count(result.get("fetched_count"))
    extracted_count = _safe_count(result.get("extracted_count"))
    failed_count = _safe_count(result.get("failed_count"))
    summary = f"已从 {source_name}同步岗位：抓取 {fetched_count} 条，写入/更新 {extracted_count} 条，失败 {failed_count} 条。"
    sync_run_id = result.get("sync_run_id")
    if sync_run_id:
        summary += f" 同步任务：{sync_run_id}。"
    return summary


def _company_database_overview_summary_response(payload: dict[str, Any]) -> str:
    tool_result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    result = tool_result.get("result") if isinstance(tool_result.get("result"), dict) else {}
    ok = payload.get("status") == "succeeded" and _tool_result_ok(tool_result)
    if not ok:
        error = payload.get("error") or tool_result.get("error") or result.get("message")
        return f"本地企业库读取失败：{error or 'unknown error'}。"

    company_count = _safe_count(result.get("company_count"))
    job_count = _safe_count(result.get("job_count"))
    job_lead_count = _safe_count(result.get("job_lead_count"))
    job_lead_company_count = _safe_count(result.get("job_lead_company_count"))
    signal_count = _safe_count(result.get("recruiting_signal_count"))
    signal_company_count = _safe_count(result.get("recruiting_signal_company_count"))
    response = (
        "可以看。我先按公司档次列出来："
    )
    table = _company_database_rows_markdown_table(result)
    if table:
        response += f"\n\n{table}\n\n"
    response += (
        "当前本地数据库里："
        f"正式企业表 {company_count} 家，正式岗位 {job_count} 条；"
        f"岗位线索 {job_lead_count} 条，去重企业 {job_lead_company_count} 家；"
        f"公司校招来源 {signal_count} 条，去重企业 {signal_company_count} 家。"
    )
    response += (
        " 这里和公司展览不是同一个统计口径：公司展览的“来源库公司数”来自当前选中的外部公司库；"
        "“当前筛选导入线索”只统计具体岗位线索；文章/社媒信号暂不计入公司数，因为这类数据字段还不完整。"
    )
    samples = _company_database_sample_text(result)
    if samples:
        response += f" 样例：{samples}。"
    response += " 后续分析和推荐可以基于这些本地企业、岗位线索和校招来源继续做。"
    return response


def _company_database_rows_markdown_table(result: dict[str, Any]) -> str:
    rows = result.get("company_rows")
    if not isinstance(rows, list) or not rows:
        return ""
    lines = ["| 档次 | 公司 | 已有信息 | 数量 | 状态 |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                _markdown_table_cell(row.get(key))
                for key in ("tier", "company_name", "known_info", "quantity", "status")
            )
            + " |"
        )
    return "\n".join(lines) if len(lines) > 2 else ""


def _markdown_table_cell(value: Any) -> str:
    return str(value or "-").replace("|", "／").replace("\r", " ").replace("\n", " ").strip() or "-"


def _company_database_sample_text(result: dict[str, Any]) -> str:
    labels = [
        ("正式企业", result.get("sample_companies")),
        ("岗位线索企业", result.get("sample_lead_companies")),
        ("校招来源企业", result.get("sample_signal_companies")),
    ]
    parts: list[str] = []
    for label, values in labels:
        if not isinstance(values, list):
            continue
        names = [str(value).strip() for value in values[:3] if str(value).strip()]
        if names:
            parts.append(f"{label}包括 {', '.join(names)}")
    return "；".join(parts)


def _job_source_overview_summary_response(payload: dict[str, Any]) -> str:
    tool_result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    result = tool_result.get("result") if isinstance(tool_result.get("result"), dict) else {}
    ok = payload.get("status") == "succeeded" and _tool_result_ok(tool_result)
    if not ok:
        error = payload.get("error") or tool_result.get("error") or result.get("message")
        return f"岗位来源读取失败：{error or 'unknown error'}。"

    source_count = _safe_count(result.get("source_count"))
    enabled_source_count = _safe_count(result.get("enabled_source_count"))
    disabled_source_count = _safe_count(result.get("disabled_source_count"))
    unsynced_source_count = _safe_count(result.get("unsynced_source_count"))
    response = (
        f"本地登记的岗位信息源共有 {source_count} 个，"
        f"其中启用 {enabled_source_count} 个，禁用 {disabled_source_count} 个，"
        f"{unsynced_source_count} 个还没有同步记录。"
    )

    external = result.get("external_job_board") if isinstance(result.get("external_job_board"), dict) else {}
    if external.get("ok"):
        openings_total = _safe_count(external.get("offerio_company_openings_total"))
        companies_total = _safe_count(external.get("offerio_company_jobs_total"))
        response += f" 公司展览当前默认外部公司库里：开放岗位公司库 {openings_total} 个，公司聚合岗位库 {companies_total} 家。"
    elif external:
        response += f" 公司展览外部公司库暂时读取失败：{external.get('error') or 'unknown error'}。"

    samples = _job_source_sample_text(result)
    if samples:
        response += f" 样例信息源：{samples}。"
    response += " 这里的“岗位来源”不是正式企业数量；如果你问的是公司展览下面的公司列表，就看开放岗位公司库和公司聚合岗位库两个外部公司库。文章/社媒信号字段不完整，暂不作为公司展示。"
    return response


def _job_source_sample_text(result: dict[str, Any]) -> str:
    sample_sources = result.get("sample_sources")
    if not isinstance(sample_sources, list):
        return ""
    parts: list[str] = []
    for source in sample_sources[:3]:
        if not isinstance(source, dict):
            continue
        name = _job_source_display_name(str(source.get("name") or "").strip())
        source_type = str(source.get("source_type") or "").strip()
        if not name:
            continue
        parts.append(f"{name}（{source_type}）" if source_type else name)
    return "，".join(parts)


def _job_source_display_name(name: str) -> str:
    return name.replace("开放岗位来源库", "开放岗位公司库")


def _apply_entry_task_summary_response(payload: dict[str, Any]) -> str:
    tool_result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    result = tool_result.get("result") if isinstance(tool_result.get("result"), dict) else {}
    ok = payload.get("status") == "succeeded" and _tool_result_ok(tool_result)
    if not ok:
        error = payload.get("error") or tool_result.get("error") or result.get("error")
        return f"申请入口外部执行任务创建失败：{error or 'unknown error'}。"

    task_id = str(result.get("task_id") or "unknown")
    envelope = result.get("task_envelope") if isinstance(result.get("task_envelope"), dict) else {}
    job = envelope.get("job") if isinstance(envelope.get("job"), dict) else {}
    company_name = str(job.get("company_name") or "目标公司")
    title = str(job.get("title") or "目标岗位")
    job_id = str(job.get("job_id") or "")
    suffix = f"（岗位 ID：{job_id}）" if job_id else ""
    dispatch = result.get("dispatch") if isinstance(result.get("dispatch"), dict) else {}
    if dispatch.get("ok") and dispatch.get("status") == "succeeded" and dispatch.get("result_status") == "found_opened":
        apply_url = str(dispatch.get("apply_url") or "")
        executor_name = str(dispatch.get("executor_name") or "外部执行 Agent")
        return (
            f"已找到申请入口：{company_name} - {title}{suffix}。入口：{apply_url}。"
            f"执行器：{executor_name}。已定位申请页并停在最终提交前，请检查页面内容后再决定是否提交。"
        )
    return (
        f"已创建申请入口外部执行任务：{task_id}。目标：{company_name} - {title}{suffix}。"
        "下一步由外部执行 Agent 打开申请页、定位投递按钮，并停在最终提交前。"
    )


def _external_web_search_summary_response(payload: dict[str, Any]) -> str:
    tool_result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    result = _external_web_search_result_payload(payload)
    if not _external_web_search_result_ok(payload):
        error = payload.get("error") or tool_result.get("error") or result.get("message")
        return f"联网搜索失败：{error or 'external web search failed'}。"
    answer = str(result.get("answer") or "").strip()
    executor_name = str(result.get("executor_name") or "外部搜索 Agent")
    if not answer:
        return f"联网搜索已由 {executor_name} 完成，但没有返回可展示的结果。"
    return answer


def _safe_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    return value


def _save_step(workflow: WorkflowRun, state: AgentState, dependencies: AgentGraphDependencies) -> None:
    workflow.current_step = state.current_step
    workflow.updated_at = utc_now()
    dependencies.checkpoint_store.save(
        workflow_run_id=workflow.id,
        checkpoint_key=state.current_step,
        state=state,
    )


def _identity_node(state: dict[str, Any]) -> dict[str, Any]:
    return state
