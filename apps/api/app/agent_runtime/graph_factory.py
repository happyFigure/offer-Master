from __future__ import annotations

import json
import logging
import re
import warnings
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import date, datetime
from typing import Any, Callable
from urllib.parse import urlparse
from uuid import uuid4

try:  # Keep LangGraph's upstream pending-deprecation noise out of targeted runtime tests.
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
except Exception:  # pragma: no cover - only relevant when langchain_core changes its warning location.
    LangChainPendingDeprecationWarning = Warning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning, module=r"langgraph\..*")

logger = logging.getLogger(__name__)

from app.agent_runtime.checkpoints import AgentCheckpointStore
from app.agent_runtime.agent_as_tool import (
    AbilityAgent,
    AgentCapabilityDefinition,
    AgentCapabilityRegistry,
    FILESYSTEM_SKILL_CAPABILITY,
    FILESYSTEM_SKILL_EXECUTOR_ID,
    LEGACY_FILESYSTEM_CAPABILITIES,
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
from app.agent_runtime.context.file_context import build_file_context_hints, extract_local_file_references
from app.agent_runtime.context.file_result_context import promote_filesystem_result_context
from app.agent_runtime.context.resource_effects import promote_contract_resource_effects_context, promote_declared_resource_effects_context
from app.agent_runtime.durable_state.service import DurableStateNotFoundError
from app.agent_runtime.goal.builder import build_goal_state
from app.agent_runtime.goal.schemas import GoalState
from app.agent_runtime.goal.validator import validate_goal_completion
from app.agent_runtime.final_answer.synthesis import complete_tool_observation_final_answer
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
from app.agent_runtime.output_sanitizer import (
    contains_false_tool_execution_claim,
    false_tool_execution_claim_fallback_response,
    sanitize_agent_final_answer,
)
from app.agent_runtime.pending_operations import (
    build_pending_operation_from_tool_input_completion,
    clear_pending_operation,
    with_pending_operation,
)
from app.agent_runtime.routing.result_envelope import build_result_envelope
from app.agent_runtime.routing.runtime_guard import validate_route_decision
from app.agent_runtime.routing.schemas import RouteDecision
from app.agent_runtime.sdk_agents.approval_bridge import (
    approval_request_create_from_waiting_payload,
    is_sdk_agent_approval_payload,
    sdk_agent_has_resumable_run_state,
    sdk_agent_approval_payload_from_result,
    sdk_agent_approval_rejected_response,
    with_sdk_agent_approval_decision_metadata,
    with_sdk_agent_approval_metadata,
)
from app.agent_runtime.sdk_agents.delegation_policy import SDK_AGENT_CHILD_MCP_PREFIXES
from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor
from app.agent_runtime.skills.filesystem_operation_catalog import FilesystemOperationSpec, get_filesystem_operation_spec
from app.agent_runtime.state import AgentState
from app.agent_runtime.tool_candidate_selector import ToolCandidateSelection
from app.agent_runtime.tool_input_completion import ToolInputCompletionResult, complete_tool_input
from app.agent_runtime.tool_input import requested_sample_limit_from_text
from app.agent_runtime.tool_permissions import AgentToolPermissionPolicy
from app.agent_runtime.tool_result_envelope import build_tool_result_envelope
from app.agent_runtime.tool_registry import (
    APPLICATION_FIND_APPLY_ENTRY_TOOL,
    DATABASE_COMPANY_LIST_TOOL,
    EXTERNAL_WEB_SEARCH_TOOL,
    LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
    LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
    OFFERIO_COMPANY_JOBS_TOOL,
    AgentToolDefinition,
    AgentToolRiskLevel,
    AgentToolRegistry,
)
from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
from app.agent_runtime.understanding.schemas import EntityFrame, IntentFrame
from app.domains.automation.models import ApprovalRequest, ApprovalRequestStatus, ToolCallStatus, WorkflowRun, WorkflowRunStatus, utc_now
from app.domains.automation.schemas import ApprovalRequestCreate, ToolCallLogCreate, WorkflowRunCreate
from app.domains.automation.service import AutomationService
from app.domains.agent_memory.repository import AgentMemoryRepository
from app.domains.conversations.models import AgentMessageRole
from app.domains.conversations.schemas import AgentMessageCreate
from app.domains.conversations.service import ConversationService
from sqlalchemy.orm import Session


AGENT_GRAPH_NODE_ORDER = ["build_context", "plan_or_reply", "maybe_tool", "wait_confirmation", "final_response"]
LOOP_RUNNER_STAGE_CONTEXT_METADATA_KEY = "loop_runner_stage_context"

# Semantic quality retry is different from transient API/network retry.
# Keep it bounded because repeated bad search rewrites can drift away from the user's intent.
DEFAULT_REFLECTION_RETRY_BUDGET = 3
MAX_REFLECTION_RETRY_BUDGET = 3
RUNTIME_CAPABILITY_APPROVAL_METADATA_KEY = "runtime_capability_approval"
RUNTIME_CAPABILITY_APPROVAL_ERROR_CODE = "RUNTIME_CAPABILITY_APPROVAL_REQUIRED"
GOAL_STATE_METADATA_KEY = "goal_state"
FILESYSTEM_COMPLETION_CONTRACT_METADATA_KEY = "filesystem_completion_contract"


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
    memory_repository: AgentMemoryRepository | None = None
    skill_repository: AgentSkillRepository | None = None
    db_session: Session | None = None
    llm_client: Any | None = None
    intent_detector: Any | None = None
    execution_planner: Any | None = None
    capability_routing_middleware: Any | None = None
    durable_state_service: Any | None = None
    agent_executors: dict[str, AbilityAgent] = field(default_factory=dict)
    capability_executor_ids: dict[str, str] = field(default_factory=dict)
    context_build_config: ContextBuildConfig = field(default_factory=ContextBuildConfig)
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


class StaleApprovalContinuationError(ValueError):
    """Raised when a UI approval card no longer matches the workflow checkpoint.

    Approval is a two-phase operation: the database approval row says what the
    user clicked, while the workflow checkpoint says where the agent is paused.
    Treating either side alone as truth can consume a stale UI card and leave the
    frontend spinning, so continuation validates both sides before marking a
    pending request approved or rejected.
    """

    error_code = "STALE_APPROVAL_REQUEST"

    def __init__(
        self,
        *,
        approval_request_id: str,
        workflow_run_id: str,
        reason: str,
        approval_status: str | None = None,
        workflow_status: str | None = None,
        workflow_current_step: str | None = None,
        workflow_approval_request_id: str | None = None,
        checkpoint_current_step: str | None = None,
        checkpoint_approval_request_id: str | None = None,
    ) -> None:
        self.approval_request_id = approval_request_id
        self.workflow_run_id = workflow_run_id
        self.reason = reason
        self.approval_status = approval_status
        self.workflow_status = workflow_status
        self.workflow_current_step = workflow_current_step
        self.workflow_approval_request_id = workflow_approval_request_id
        self.checkpoint_current_step = checkpoint_current_step
        self.checkpoint_approval_request_id = checkpoint_approval_request_id
        self.message = "这个确认请求已经过期或任务状态已变化，请刷新会话后重新发起。"
        super().__init__(self.message)

    def to_payload(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "reason": self.reason,
            "approval_request_id": self.approval_request_id,
            "workflow_run_id": self.workflow_run_id,
            "approval_status": self.approval_status,
            "workflow_status": self.workflow_status,
            "workflow_current_step": self.workflow_current_step,
            "workflow_approval_request_id": self.workflow_approval_request_id,
            "checkpoint_current_step": self.checkpoint_current_step,
            "checkpoint_approval_request_id": self.checkpoint_approval_request_id,
        }


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


def _state_is_waiting_for_user(state: AgentState) -> bool:
    return state.current_step in {"wait_confirmation", "wait_user_input"}


def _approval_request_create_for_waiting_state(
    workflow: WorkflowRun,
    state: AgentState,
    *,
    fallback_action_type: str,
    fallback_prompt: str,
    fallback_payload: dict[str, Any],
) -> ApprovalRequestCreate:
    runtime_payload = _runtime_capability_approval_payload_from_state_metadata(state.context_metadata)
    if runtime_payload is not None:
        return ApprovalRequestCreate(
            workflow_run_id=workflow.id,
            action_type=str(runtime_payload.get("requested_tool_name") or fallback_action_type),
            prompt=str(runtime_payload.get("user_message") or runtime_payload.get("reason") or fallback_prompt),
            payload={**runtime_payload, "outer_approval_payload": dict(fallback_payload)},
        )
    return approval_request_create_from_waiting_payload(
        workflow_run_id=workflow.id,
        fallback_action_type=fallback_action_type,
        fallback_prompt=fallback_prompt,
        fallback_payload=fallback_payload,
        state_metadata=state.context_metadata,
    )


def run_agent_workflow(
    command: AgentRunCommand,
    *,
    dependencies: AgentGraphDependencies,
    on_workflow_started: Callable[[WorkflowRun, AgentState], AgentState | None] | None = None,
) -> AgentWorkflowResult:
    prepared = prepare_agent_workflow_response(
        command,
        dependencies=dependencies,
        on_workflow_started=on_workflow_started,
    )
    if _state_is_waiting_for_user(prepared.state):
        return AgentWorkflowResult(workflow_run_id=prepared.workflow_run_id, state=prepared.state)

    final_response, response_mode = _generate_final_response(prepared.state, dependencies=dependencies)
    prepared_state = prepared.state
    if _state_is_waiting_for_user(prepared_state):
        return AgentWorkflowResult(workflow_run_id=prepared.workflow_run_id, state=prepared_state)
    return finalize_agent_workflow_response(
        prepared_state,
        final_response=final_response,
        response_mode=response_mode,
        dependencies=dependencies,
    )


def prepare_agent_workflow_response(
    command: AgentRunCommand,
    *,
    dependencies: AgentGraphDependencies,
    on_workflow_started: Callable[[WorkflowRun, AgentState], AgentState | None] | None = None,
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
        dependencies=dependencies,
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
    dependencies: AgentGraphDependencies,
) -> tuple[AgentState, str, str]:
    sanitized = sanitize_agent_final_answer(final_response)
    if not sanitized.removed_internal_protocol:
        return _sanitize_false_tool_execution_claim_for_user(state, final_response=final_response, response_mode=response_mode)

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
        return _sanitize_false_tool_execution_claim_for_user(state, final_response=sanitized.content, response_mode=response_mode)

    tool_response = tool_result_summary_response(state, dependencies=dependencies)
    if tool_response is not None:
        fallback_content, fallback_mode = tool_response
        return state, fallback_content, fallback_mode
    return state, "我已完成处理，但最终回答需要重新整理。请重新发送问题或换一种问法。", "sanitized_empty_fallback"


def _sanitize_false_tool_execution_claim_for_user(
    state: AgentState,
    *,
    final_response: str,
    response_mode: str,
) -> tuple[AgentState, str, str]:
    if state.tool_call_ids or not contains_false_tool_execution_claim(final_response):
        return state, final_response, response_mode
    existing_sanitizer = state.context_metadata.get("output_sanitizer") if isinstance(state.context_metadata, dict) else None
    metadata = {
        **state.context_metadata,
        "output_sanitizer": {
            **(dict(existing_sanitizer) if isinstance(existing_sanitizer, dict) else {}),
            "removed_false_tool_claim": True,
            "needs_regeneration": True,
        },
    }
    return state.with_updates(context_metadata=metadata), false_tool_execution_claim_fallback_response(), "false_tool_claim_fallback"

def resume_agent_workflow(workflow_run_id: str, *, dependencies: AgentGraphDependencies) -> AgentWorkflowResult:
    snapshot = dependencies.checkpoint_store.load_latest(workflow_run_id)
    return AgentWorkflowResult(workflow_run_id=workflow_run_id, state=snapshot.state)


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _ensure_approval_can_continue(approval: ApprovalRequest, snapshot, workflow: WorkflowRun) -> None:
    approval_status = _enum_value(approval.status)
    workflow_status = _enum_value(workflow.status)
    checkpoint_approval_request_id = str(snapshot.state.approval_request_id or "") or None
    workflow_approval_request_id = str(workflow.approval_request_id or "") or None

    def raise_stale(reason: str) -> None:
        raise StaleApprovalContinuationError(
            approval_request_id=approval.id,
            workflow_run_id=approval.workflow_run_id,
            reason=reason,
            approval_status=approval_status,
            workflow_status=workflow_status,
            workflow_current_step=workflow.current_step,
            workflow_approval_request_id=workflow_approval_request_id,
            checkpoint_current_step=snapshot.state.current_step,
            checkpoint_approval_request_id=checkpoint_approval_request_id,
        )

    if approval_status != ApprovalRequestStatus.PENDING.value:
        raise_stale("approval_request_not_pending")
    if workflow_status != WorkflowRunStatus.WAITING_USER.value:
        raise_stale("workflow_not_waiting_user")
    if workflow.current_step != "wait_confirmation" or snapshot.state.current_step != "wait_confirmation":
        raise_stale("workflow_not_wait_confirmation")
    if workflow_approval_request_id != approval.id or checkpoint_approval_request_id != approval.id:
        raise_stale("approval_request_id_mismatch")


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

    snapshot = dependencies.checkpoint_store.load_latest(approval.workflow_run_id)
    workflow = dependencies.db_session.get(WorkflowRun, approval.workflow_run_id)
    if workflow is None:
        raise ValueError(f"Workflow run not found: {approval.workflow_run_id}")

    _ensure_approval_can_continue(approval, snapshot, workflow)
    approval = dependencies.automation_service.decide_approval(
        approval_request_id,
        approved=approved,
        decision=decision_reason,
    )

    payload = approval.payload or {}
    sdk_approval_payload = payload if is_sdk_agent_approval_payload(payload) else None

    if sdk_approval_payload is not None and sdk_agent_has_resumable_run_state(sdk_approval_payload):
        # Real SDK interruptions preserve RunState; resume the child agent instead of replaying its inner tool outside the SDK.
        state = _resume_sdk_agent_after_approval(
            snapshot.state,
            workflow=workflow,
            approval=approval,
            approved=approved,
            decision_reason=decision_reason,
            sdk_approval_payload=sdk_approval_payload,
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

    if not approved:
        context_metadata = dict(snapshot.state.context_metadata)
        final_response = "User rejected the pending tool call. The Agent stopped before executing the tool."
        if sdk_approval_payload is not None:
            # SDK sub-agent approvals carry the inner tool request; preserve that decision for audit.
            context_metadata = with_sdk_agent_approval_decision_metadata(
                context_metadata,
                sdk_approval_payload,
                approved=False,
                decision_reason=decision_reason,
            )
            final_response = sdk_agent_approval_rejected_response(sdk_approval_payload, decision_reason=decision_reason)
        state = snapshot.state.with_updates(
            current_step="approval_rejected",
            approval_request_id=approval.id,
            final_response=final_response,
            response_mode="user_rejected",
            context_metadata=context_metadata,
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
        workflow.status = WorkflowRunStatus.CANCELED
        workflow.current_step = state.current_step
        workflow.completed_at = utc_now()
        _save_step(workflow, state, dependencies)
        return AgentWorkflowResult(workflow_run_id=workflow.id, state=state)

    if snapshot.state.current_step != "wait_confirmation":
        raise ValueError(f"Workflow is not waiting for user confirmation: {approval.workflow_run_id}")

    requested_tool_name = str(payload.get("requested_tool_name") or approval.action_type or snapshot.state.requested_tool_name or "")
    source_type = str(payload.get("source_type") or snapshot.state.source_type or "agent_chat")
    tool_input = payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
    context_metadata = _approval_continuation_context_metadata(snapshot.state.context_metadata, tool_input)
    if sdk_approval_payload is not None:
        # Resume from the SDK approval payload so approval executes only the requested inner tool once.
        context_metadata = with_sdk_agent_approval_decision_metadata(
            context_metadata,
            sdk_approval_payload,
            approved=True,
            decision_reason=decision_reason,
        )
    state = snapshot.state.with_updates(
        current_step="maybe_tool",
        approval_request_id=approval.id,
        requested_tool_name=requested_tool_name,
        source_type=source_type,
        context_metadata=context_metadata,
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


_APPROVAL_CONTINUATION_CONTEXT_KEYS = (
    GOAL_STATE_METADATA_KEY,
    "active_file",
    "active_directory",
    "recent_directory_paths",
    "filesystem_operation",
    "operation_intent",
    "pending_operation",
    "resource_effects",
    "active_resource",
    "artifact_context",
    "last_file_operation_result",
    "context_pack",
    "intent_frame",
)


def _approval_continuation_context_metadata(base_metadata: dict[str, Any], tool_input: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(base_metadata)
    tool_context = tool_input.get("context_metadata") if isinstance(tool_input.get("context_metadata"), dict) else None
    if not isinstance(tool_context, dict):
        return metadata
    for key in _APPROVAL_CONTINUATION_CONTEXT_KEYS:
        if key not in metadata and key in tool_context:
            # Approval payloads are server-created checkpoints. Rehydrating only
            # these intent/file frame keys prevents the confirmation round-trip
            # from losing the exact user goal the tool result must satisfy.
            metadata[key] = tool_context[key]
    return metadata


def _resume_sdk_agent_after_approval(
    state: AgentState,
    *,
    workflow: WorkflowRun,
    approval: ApprovalRequest,
    approved: bool,
    decision_reason: str | None,
    sdk_approval_payload: dict[str, Any],
    dependencies: AgentGraphDependencies,
) -> AgentState:
    outer_capability = str(sdk_approval_payload.get("outer_capability") or approval.action_type or state.requested_tool_name or "")
    outer_tool_input = sdk_approval_payload.get("outer_tool_input") if isinstance(sdk_approval_payload.get("outer_tool_input"), dict) else {}
    executor_id = str(sdk_approval_payload.get("executor_id") or _agent_runtime_executor_id(AgentRunCommand(state.session_id, state.user_message, outer_capability), dependencies=dependencies))
    source_type = str(sdk_approval_payload.get("source_type") or state.source_type or "agent_chat")
    command = AgentRunCommand(
        session_id=state.session_id,
        user_message=state.user_message,
        requested_tool_name=outer_capability,
        source_type=source_type,
        user_confirmed=approved,
        tool_input=dict(outer_tool_input),
    )
    context_metadata = with_sdk_agent_approval_decision_metadata(
        state.context_metadata,
        sdk_approval_payload,
        approved=approved,
        decision_reason=decision_reason,
    )
    state = state.with_updates(
        current_step="maybe_tool",
        approval_request_id=approval.id,
        requested_tool_name=outer_capability,
        source_type=source_type,
        context_metadata=context_metadata,
    )
    durable_step_id = _begin_durable_tool_step(
        state,
        command=command,
        dependencies=dependencies,
        tool_input=dict(outer_tool_input),
    )
    guard_payload = {
        "ok": True,
        "error_code": None,
        "reason": "SDK agent approval decision accepted.",
        "user_message": None,
        "next_action": "execute",
        "retryable": False,
        "error_details": {
            "executor_id": executor_id,
            "sdk_agent_approval_resume": True,
            "approved": bool(approved),
            "approval_request_id": approval.id,
            "requested_tool_name": sdk_approval_payload.get("requested_tool_name"),
        },
        "cost": {},
        "artifacts": {},
    }
    resume = _sdk_agent_resume_method(sdk_approval_payload, dependencies=dependencies)
    if resume is None:
        agent_result = StandardAgentResult(
            status="failed",
            summary="SDK approval payload has RunState, but no SDK executor can resume it.",
            raw_result={"tool_name": outer_capability, "ok": False, "error_type": "SdkRunStateResumeUnavailable"},
        )
    else:
        try:
            agent_result = resume(
                sdk_approval_payload,
                approved=approved,
                context=AgentRuntimeContext(
                    session_id=state.session_id,
                    run_id=state.workflow_run_id,
                    task_id=f"{state.workflow_run_id}:sdk-resume-{len(state.tool_call_ids) + 1}",
                    namespace=executor_id,
                    permission_scope={
                        "source_type": source_type,
                        "user_confirmed": approved,
                        "sdk_agent_approval_resume": True,
                    },
                    metadata={
                        "agent_run_id": state.agent_run_id,
                        "approval_request_id": approval.id,
                        "decision_reason": decision_reason,
                    },
                    capability_id=outer_capability,
                    event_sink=dependencies.event_sink,
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive boundary for external SDK adapters.
            agent_result = StandardAgentResult(
                status="failed",
                summary=f"SDK RunState resume failed: {type(exc).__name__}: {exc}",
                raw_result={"tool_name": outer_capability, "ok": False, "error_type": type(exc).__name__, "error": str(exc)},
            )

    next_sdk_approval_payload = sdk_agent_approval_payload_from_result(
        agent_result,
        outer_capability=outer_capability,
        outer_tool_input=dict(outer_tool_input),
        executor_id=executor_id,
    )
    if next_sdk_approval_payload is not None:
        return _pause_sdk_agent_resume_for_approval(
            state,
            workflow=workflow,
            command=command,
            agent_result=agent_result,
            sdk_approval_payload=next_sdk_approval_payload,
            context_metadata=context_metadata,
            outer_tool_input=dict(outer_tool_input),
            executor_id=executor_id,
            durable_step_id=durable_step_id,
            dependencies=dependencies,
        )

    result_payload = _jsonable(agent_result.raw_result)
    if not result_payload:
        result_payload = {"tool_name": outer_capability, "ok": agent_result.status != "failed", "result": {"summary": agent_result.summary}}
    result_payload = _with_result_envelope(outer_capability, result_payload, state=state)
    tool_ok = agent_result.status != "failed" and _tool_result_ok(result_payload)
    tool_error = None if tool_ok else _tool_result_error(result_payload)
    if tool_ok:
        # A resumed SDK child agent may create files after user approval. Promote
        # those declared effects into runtime context so follow-up turns can
        # resolve phrases like "刚才生成的文件" without re-reading chat logs.
        context_metadata = promote_contract_resource_effects_context(
            context_metadata,
            result_payload,
            tool_input=dict(outer_tool_input),
            semantic_profile=_semantic_profile_for_tool(dependencies.registry, outer_capability),
        )
        context_metadata = promote_declared_resource_effects_context(context_metadata, result_payload)
        context_metadata = promote_filesystem_result_context(context_metadata, result_payload)
    tool_call = dependencies.automation_service.record_tool_call(
        ToolCallLogCreate(
            workflow_run_id=state.workflow_run_id,
            tool_name=outer_capability,
            tool_group="agent",
            status=ToolCallStatus.SUCCEEDED if tool_ok else ToolCallStatus.FAILED,
            input_payload=dict(outer_tool_input),
            output_payload={
                "guard_result": guard_payload,
                "execution": "sdk_agent_resume",
                "agent_runtime": _agent_runtime_result_metadata(agent_result, executor_id=executor_id),
                "result": result_payload,
                "sdk_agent_approval_decision": context_metadata.get("sdk_agent_approval_decision"),
            },
            error=tool_error,
        )
    )
    _mark_durable_tool_step_completed(
        durable_step_id,
        state=state,
        command=command,
        tool_input=dict(outer_tool_input),
        dependencies=dependencies,
        succeeded=tool_ok,
        tool_call_log_id=tool_call.id,
        external_task_id=_extract_external_task_id(result_payload),
        output_payload={
            "guard_result": guard_payload,
            "execution": "sdk_agent_resume",
            "agent_runtime": _agent_runtime_result_metadata(agent_result, executor_id=executor_id),
            "result": result_payload,
            "error": tool_error,
        },
    )
    tool_messages = _append_tool_pair_messages(
        state,
        dependencies=dependencies,
        tool_call_log_id=tool_call.id,
        tool_name=outer_capability,
        tool_input=dict(outer_tool_input),
        status="succeeded" if tool_ok else "failed",
        result=result_payload,
        error=tool_error,
    )
    _record_skill_runtime_event(
        state,
        dependencies=dependencies,
        event="tool_succeeded" if tool_ok else "tool_failed",
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
        tool_input=dict(outer_tool_input),
        tool_call_id=tool_call.id,
        status="succeeded" if tool_ok else "failed",
        summary=agent_result.summary,
    )
    return state.with_updates(
        current_step="maybe_tool",
        guard_result=guard_payload,
        tool_call_ids=[*state.tool_call_ids, tool_call.id],
        llm_messages=[*state.llm_messages, *tool_messages],
        context_metadata=context_metadata,
        final_response=agent_result.summary,
        response_mode="sdk_agent_resume",
    )


def _pause_sdk_agent_resume_for_approval(
    state: AgentState,
    *,
    workflow: WorkflowRun,
    command: AgentRunCommand,
    agent_result: StandardAgentResult,
    sdk_approval_payload: dict[str, Any],
    context_metadata: dict[str, Any],
    outer_tool_input: dict[str, Any],
    executor_id: str,
    durable_step_id: str | None,
    dependencies: AgentGraphDependencies,
) -> AgentState:
    guard_payload = sdk_approval_payload["guard_result"]
    result_payload = _jsonable(agent_result.raw_result)
    tool_error = f"{guard_payload.get('error_code')}: {guard_payload.get('reason')}"
    tool_call = dependencies.automation_service.record_tool_call(
        ToolCallLogCreate(
            workflow_run_id=state.workflow_run_id,
            tool_name=command.requested_tool_name or "",
            tool_group="agent",
            status=ToolCallStatus.BLOCKED,
            input_payload=dict(outer_tool_input),
            output_payload={
                "guard_result": guard_payload,
                "execution": "sdk_agent_resume_approval",
                "agent_runtime": _agent_runtime_result_metadata(agent_result, executor_id=executor_id),
                "result": result_payload,
                "sdk_agent_approval": sdk_approval_payload,
            },
            error=tool_error,
        )
    )
    _mark_durable_tool_step_waiting_user(
        durable_step_id,
        state=state,
        command=command,
        tool_input=dict(outer_tool_input),
        dependencies=dependencies,
        tool_call_log_id=tool_call.id,
        output_payload={
            "guard_result": guard_payload,
            "execution": "sdk_agent_resume_approval",
            "agent_runtime": _agent_runtime_result_metadata(agent_result, executor_id=executor_id),
            "result": result_payload,
            "error": tool_error,
            "sdk_agent_approval": sdk_approval_payload,
        },
    )
    tool_messages = _append_tool_pair_messages(
        state,
        dependencies=dependencies,
        tool_call_log_id=tool_call.id,
        tool_name=command.requested_tool_name or "",
        tool_input=dict(outer_tool_input),
        status="waiting_user",
        result=result_payload,
        error=tool_error,
    )
    waiting_metadata = with_sdk_agent_approval_metadata(context_metadata, sdk_approval_payload)
    waiting_state = state.with_updates(
        current_step="wait_confirmation",
        guard_result=guard_payload,
        tool_call_ids=[*state.tool_call_ids, tool_call.id],
        llm_messages=[*state.llm_messages, *tool_messages],
        context_metadata=waiting_metadata,
        final_response=str(guard_payload.get("user_message") or guard_payload.get("reason") or agent_result.summary),
        response_mode="sdk_agent_approval",
    )
    # A resumed SDK run can pause again; create a fresh approval tied to the inner request it just surfaced.
    approval = dependencies.automation_service.request_user_approval(
        _approval_request_create_for_waiting_state(
            workflow,
            waiting_state,
            fallback_action_type=str(sdk_approval_payload.get("requested_tool_name") or command.requested_tool_name or "sdk_agent_tool_call"),
            fallback_prompt=str(sdk_approval_payload.get("user_message") or sdk_approval_payload.get("reason") or "Confirm SDK agent tool call."),
            fallback_payload={
                "agent_run_id": waiting_state.agent_run_id,
                "source_type": command.source_type,
                "requested_tool_name": sdk_approval_payload.get("requested_tool_name") or command.requested_tool_name,
                "tool_input": sdk_approval_payload.get("tool_input") if isinstance(sdk_approval_payload.get("tool_input"), dict) else {},
                "guard_result": guard_payload,
            },
        )
    ).approval
    workflow.status = WorkflowRunStatus.WAITING_USER
    workflow.current_step = "wait_confirmation"
    return waiting_state.with_updates(approval_request_id=approval.id)


def _sdk_agent_resume_method(
    sdk_approval_payload: dict[str, Any],
    *,
    dependencies: AgentGraphDependencies,
) -> Callable[..., StandardAgentResult] | None:
    executor_id = str(sdk_approval_payload.get("executor_id") or "")
    executor = dependencies.agent_executors.get(executor_id)
    resume = getattr(executor, "resume_after_approval", None)
    return resume if callable(resume) else None


def _run_until_response_ready(
    state: AgentState,
    *,
    command: AgentRunCommand,
    workflow: WorkflowRun,
    dependencies: AgentGraphDependencies,
    on_workflow_started: Callable[[WorkflowRun, AgentState], AgentState | None] | None = None,
) -> AgentPreparedResponse:
    # Declared Agent capabilities must be present before ContextPack is built,
    # otherwise normal_chat and unrelated intents cannot delegate to MCP agents.
    dependencies = _dependencies_with_declared_agent_capability_tools(dependencies)
    state = _build_context_node(state, dependencies=dependencies)
    _record_durable_context_snapshots(state, dependencies=dependencies)
    _save_step(workflow, state, dependencies)
    if on_workflow_started is not None:
        next_state = on_workflow_started(workflow, state)
        if next_state is not None:
            state = next_state

    command = _auto_select_tool_command(command, state=state, registry=dependencies.registry)
    if command.requested_tool_name:
        state = state.with_updates(requested_tool_name=command.requested_tool_name, source_type=command.source_type)

    state = state.with_updates(current_step="plan_or_reply")
    _save_step(workflow, state, dependencies)

    if not command.requested_tool_name:
        # The main Agent is the first semantic decision maker. It receives the
        # server-loaded history and the complete source-eligible capability
        # catalog, then submits a structured call when work is needed.
        state = _tool_choice_loop_node(state, dependencies=dependencies)
        if state.current_step == "wait_confirmation":
            approval = dependencies.automation_service.request_user_approval(
                _approval_request_create_for_waiting_state(
                    workflow,
                    state,
                    fallback_action_type=state.requested_tool_name or "tool_choice_call",
                    fallback_prompt=f"Confirm before running tool: {state.requested_tool_name}",
                    fallback_payload={
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
        if state.current_step == "wait_user_input":
            workflow.status = WorkflowRunStatus.WAITING_USER
            workflow.current_step = "wait_user_input"
            _save_step(workflow, state, dependencies)
            return AgentPreparedResponse(workflow_run_id=workflow.id, workflow=workflow, state=state)
        if state.final_response and _has_prepared_final_response(state):
            _save_step(workflow, state, dependencies)
            return AgentPreparedResponse(workflow_run_id=workflow.id, workflow=workflow, state=state)

    if command.requested_tool_name:
        state = _maybe_tool_node(state, command=command, dependencies=dependencies)
        if state.current_step == "maybe_tool":
            state = _recover_explicit_tool_no_dead_end(
                state,
                command=command,
                dependencies=dependencies,
            )
        if state.current_step == "wait_confirmation":
            approval = dependencies.automation_service.request_user_approval(
                _approval_request_create_for_waiting_state(
                    workflow,
                    state,
                    fallback_action_type=command.requested_tool_name or "tool_call",
                    fallback_prompt=f"Confirm before running tool: {command.requested_tool_name}",
                    fallback_payload={
                        "agent_run_id": state.agent_run_id,
                        "source_type": command.source_type,
                        "requested_tool_name": command.requested_tool_name,
                        "tool_input": _pending_runtime_tool_input(state) or command.tool_input,
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
        if state.current_step == "wait_user_input":
            workflow.status = WorkflowRunStatus.WAITING_USER
            workflow.current_step = "wait_user_input"
            _save_step(workflow, state, dependencies)
            return AgentPreparedResponse(workflow_run_id=workflow.id, workflow=workflow, state=state)
        _save_step(workflow, state, dependencies)

    return AgentPreparedResponse(workflow_run_id=workflow.id, workflow=workflow, state=state)


def _recover_explicit_tool_no_dead_end(
    state: AgentState,
    *,
    command: AgentRunCommand,
    dependencies: AgentGraphDependencies,
) -> AgentState:
    """Continue an explicit capability call when its result exposes recovery work."""

    requested_tool_name = str(state.requested_tool_name or command.requested_tool_name or "").strip()
    if not requested_tool_name:
        return state
    original_input = _pending_runtime_tool_input(state) or dict(command.tool_input)
    payload = _latest_tool_result_payload(state, requested_tool_name)
    recovery_outcome = _no_dead_end_outcome_from_payload(payload)
    logger.info(
        "Explicit capability recovery inspection: capability=%s payload_keys=%s recovery_keys=%s current_step=%s",
        requested_tool_name,
        sorted(payload.keys()) if isinstance(payload, dict) else [],
        sorted(recovery_outcome.keys()) if isinstance(recovery_outcome, dict) else [],
        state.current_step,
    )
    if (
        isinstance(recovery_outcome, dict)
        and str(recovery_outcome.get("next_action") or "").strip() == "continue_model_loop"
        and dependencies.llm_client is not None
    ):
        # A coarse capability call can legitimately fail because the model has
        # not yet selected the inner filesystem operation or filename intent.
        # Feed that structured observation into the same native tool loop so
        # the model gets another decision turn; never infer the missing field
        # from the user's prose here.
        logger.info(
            "Continuing model loop after recoverable explicit capability result: capability=%s error_code=%s",
            requested_tool_name,
            recovery_outcome.get("error_code"),
        )
        if (
            requested_tool_name == FILESYSTEM_SKILL_CAPABILITY
            and not str(original_input.get("operation") or "").strip()
        ):
            state = _arm_filesystem_write_follow_up_candidate(state, original_tool_input=original_input)
        return _tool_choice_loop_node(
            state,
            dependencies=dependencies,
            capabilities_override=(requested_tool_name,),
        )
    decision = _no_dead_end_recovery_decision_from_payload(requested_tool_name, original_input, payload)
    if decision is None or decision.action != LoopAgentAction.CALL_TOOL:
        return state

    recovery_capability = str(decision.capability or requested_tool_name).strip()
    recovery_input = dict(decision.tool_input or {})
    recovery_state = _maybe_tool_node(
        state.with_updates(requested_tool_name=recovery_capability, source_type="agent_chat"),
        command=AgentRunCommand(
            session_id=state.session_id,
            user_message=state.user_message,
            requested_tool_name=recovery_capability,
            source_type="agent_chat",
            user_confirmed=False,
            tool_input=recovery_input,
        ),
        dependencies=dependencies,
    )
    if recovery_state.current_step in {"wait_confirmation", "wait_user_input"}:
        return recovery_state

    # Once read-before-write evidence is available, let the model choose the
    # concrete destination through the bounded loop that preserves the actual
    # tool observations in state.llm_messages. The separate candidate loop
    # rebuilds messages from the user prompt and can otherwise answer from
    # stale intent without issuing the required mutation call.
    if recovery_state.tool_call_ids and dependencies.llm_client is not None:
        recovery_state = _mark_filesystem_write_follow_up_contract(
            recovery_state,
            original_tool_input=original_input,
            recovery_outcome=recovery_outcome,
        )
        return _tool_choice_loop_node(
            recovery_state,
            dependencies=dependencies,
            capabilities_override=(recovery_capability,),
        )
    return recovery_state


def _mark_filesystem_write_follow_up_contract(
    state: AgentState,
    *,
    original_tool_input: dict[str, Any] | None = None,
    recovery_outcome: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> AgentState:
    """Persist a runtime-owned read-before-write continuation contract.

    The contract is created only from structured runtime/tool evidence. It does
    not parse the user's prose and it never supplies a filename or destination.
    """

    original_tool_input = original_tool_input if isinstance(original_tool_input, dict) else {}
    recovery_outcome = recovery_outcome if isinstance(recovery_outcome, dict) else {}
    payload = payload or _latest_tool_result_payload(state, FILESYSTEM_SKILL_CAPABILITY) or {}
    operation = str(original_tool_input.get("operation") or "").strip()
    read_operation = _filesystem_payload_operation(payload)
    inferred_recovery = recovery_outcome or _no_dead_end_outcome_from_payload(payload) or {}
    next_action = str(
        recovery_outcome.get("next_action")
        or inferred_recovery.get("next_action")
        or ""
    ).strip()

    # A coarse filesystem capability can fail before the model selects the
    # inner operation. Preserve that structured recovery signal immediately so
    # the same native loop remains responsible for the later read and write.
    # This uses only runtime-owned fields and never invents a filename or dst.
    if (
        not operation
        and not read_operation
        and next_action == "continue_model_loop"
        and isinstance(inferred_recovery, dict)
    ):
        logger.info(
            "Filesystem completion candidate armed after coarse operation recovery: error_code=%s source=%s",
            inferred_recovery.get("error_code"),
            original_tool_input.get("path") or original_tool_input.get("src"),
        )
        return _arm_filesystem_write_follow_up_candidate(state, original_tool_input=original_tool_input)
    existing = state.context_metadata.get(FILESYSTEM_COMPLETION_CONTRACT_METADATA_KEY)
    if (
        read_operation == "read_file"
        and _tool_result_ok(payload)
        and isinstance(existing, dict)
        and existing.get("requires_follow_up_after_read")
        and not existing.get("awaiting_write")
    ):
        source_path = str(
            existing.get("source_path")
            or original_tool_input.get("src")
            or original_tool_input.get("path")
            or _filesystem_payload_argument(payload, "path")
            or _filesystem_payload_argument(payload, "src")
            or ""
        ).strip()
        contract = {
            **existing,
            "awaiting_write": True,
            "requires_follow_up_after_read": False,
            "source_path": source_path,
            "source": "runtime_read_before_write",
        }
        logger.info("Filesystem completion contract upgraded after verified read: source=%s", source_path)
        return state.with_updates(
            context_metadata={
                **state.context_metadata,
                FILESYSTEM_COMPLETION_CONTRACT_METADATA_KEY: contract,
            }
        )
    if read_operation in {"copy_file", "rename_file"} and _tool_result_ok(payload):
        if not isinstance(existing, dict):
            return state
        logger.info(
            "Filesystem completion contract cleared after verified mutation: operation=%s",
            read_operation,
        )
        return state.with_updates(
            context_metadata={
                key: value
                for key, value in state.context_metadata.items()
                if key != FILESYSTEM_COMPLETION_CONTRACT_METADATA_KEY
            }
        )
    if operation not in {"copy_file", "rename_file"} and next_action != "read_before_write":
        return state
    if read_operation != "read_file":
        return state
    if isinstance(existing, dict) and existing.get("awaiting_write"):
        return state

    source_path = str(
        original_tool_input.get("src")
        or original_tool_input.get("path")
        or _filesystem_payload_argument(payload, "path")
        or _filesystem_payload_argument(payload, "src")
        or ""
    ).strip()
    contract = {
        "awaiting_write": True,
        "capability": FILESYSTEM_SKILL_CAPABILITY,
        "source_path": source_path,
        "operation_hint": operation if operation in {"copy_file", "rename_file"} else None,
        "source": "runtime_read_before_write",
    }
    logger.info(
        "Filesystem completion contract armed: source=%s operation_hint=%s next_action=%s",
        source_path,
        contract["operation_hint"],
        next_action,
    )
    return state.with_updates(
        context_metadata={
            **state.context_metadata,
            FILESYSTEM_COMPLETION_CONTRACT_METADATA_KEY: contract,
        }
    )


def _arm_filesystem_write_follow_up_candidate(
    state: AgentState,
    *,
    original_tool_input: dict[str, Any] | None = None,
) -> AgentState:
    """Remember a coarse filesystem recovery that must be followed through."""

    existing = state.context_metadata.get(FILESYSTEM_COMPLETION_CONTRACT_METADATA_KEY)
    if isinstance(existing, dict) and (existing.get("awaiting_write") or existing.get("requires_follow_up_after_read")):
        return state
    original_tool_input = original_tool_input if isinstance(original_tool_input, dict) else {}
    source_path = str(original_tool_input.get("src") or original_tool_input.get("path") or "").strip()
    contract = {
        "awaiting_write": False,
        "requires_follow_up_after_read": True,
        "capability": FILESYSTEM_SKILL_CAPABILITY,
        "source_path": source_path,
        "source": "runtime_structured_operation_recovery",
    }
    logger.info("Filesystem write follow-up candidate armed: source=%s", source_path)
    return state.with_updates(
        context_metadata={
            **state.context_metadata,
            FILESYSTEM_COMPLETION_CONTRACT_METADATA_KEY: contract,
        }
    )


def _filesystem_payload_operation(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""
    candidates: list[Any] = [payload.get("operation")]
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    nested = result.get("result") if isinstance(result.get("result"), dict) else {}
    candidates.extend([result.get("operation"), nested.get("operation")])
    for candidate in candidates:
        value = str(candidate or "").strip()
        if value:
            return value
    return ""


def _filesystem_payload_argument(payload: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(payload, dict):
        return None
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    nested = result.get("result") if isinstance(result.get("result"), dict) else {}
    for candidate in (payload, result, nested):
        arguments = candidate.get("arguments") if isinstance(candidate, dict) else None
        if isinstance(arguments, dict) and arguments.get(key):
            return arguments.get(key)
        if isinstance(candidate, dict) and candidate.get(key):
            return candidate.get(key)
    return None


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
    # Route decisions can target high-level runtime capabilities such as
    # skill.filesystem, not just legacy ToolRegistry entries. Validate against
    # the same capability registry used by actual runtime execution.
    return validate_route_decision(
        decision,
        context_pack=context_pack,
        registry=_runtime_capability_registry(dependencies),
    ).to_metadata_dict()


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


def _should_offer_tool_choice_before_specialized_routing(state: AgentState) -> bool:
    """Let confident specialist intents reach their route before broad tool selection.

    Filesystem stays in the model loop because its inner operation is model-owned;
    other confident intents already have a dedicated planner/runtime route. Generic
    or uncertain requests still offer always-available child Agents to the model.
    """
    metadata = state.context_metadata if isinstance(state.context_metadata, dict) else {}
    intent_frame = metadata.get("intent_frame") if isinstance(metadata.get("intent_frame"), dict) else {}
    context_pack = metadata.get("context_pack") if isinstance(metadata.get("context_pack"), dict) else {}
    # An explicit capability is a runtime contract, not a suggestion for the
    # broad model tool chooser. Let the capability router handle it first;
    # otherwise a model-selected Chrome/DB tool can steal the request before
    # the required DBX fail-closed check runs.
    if str(intent_frame.get("required_capability") or context_pack.get("required_capability") or "").strip():
        return False
    intent = str(intent_frame.get("intent") or "normal_chat")
    try:
        confidence = float(intent_frame.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return intent in {"normal_chat", "filesystem_operation"} or confidence < 0.75


def _tool_choice_loop_node(
    state: AgentState,
    *,
    dependencies: AgentGraphDependencies,
    capabilities_override: tuple[str, ...] | None = None,
) -> AgentState:
    runtime_dependencies = _dependencies_with_declared_agent_capability_tools(dependencies)
    capability_registry = _runtime_capability_registry(runtime_dependencies)
    selection = _tool_choice_loop_candidate_selection(state, dependencies=runtime_dependencies)
    selection_metadata = selection.to_metadata_dict()
    state = state.with_updates(
        context_metadata=_with_runtime_decision_trace(
            state.context_metadata,
            "tool_candidate_selection",
            selection_metadata,
        )
    )
    capabilities = tuple(capabilities_override or selection.capabilities)
    if not capabilities:
        return state

    # A recovered coarse filesystem call has already proven that this loop is
    # continuing an incomplete delegated filesystem task. Preserve that fact
    # before the model chooses read_file, so a read observation cannot be
    # mistaken for final completion.
    if (
        capabilities_override == (FILESYSTEM_SKILL_CAPABILITY,)
        and state.tool_call_ids
        and not isinstance(state.context_metadata.get(FILESYSTEM_COMPLETION_CONTRACT_METADATA_KEY), dict)
    ):
        state = _arm_filesystem_write_follow_up_candidate(state)

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
        if next_state.current_step == "wait_user_input":
            guard_result = next_state.guard_result if isinstance(next_state.guard_result, dict) else {}
            return LoopAgentObservation(
                status="waiting_user",
                summary=str(guard_result.get("user_message") or guard_result.get("reason") or "工具参数还不完整，请补充后继续。"),
                requires_user_action=True,
                metadata={"guard_result": guard_result, "tool_input": tool_input},
            )
        if len(next_state.tool_call_ids) <= before_tool_call_count:
            guard_result = next_state.guard_result if isinstance(next_state.guard_result, dict) else {}
            return LoopAgentObservation(
                status="failed",
                summary=str(guard_result.get("user_message") or guard_result.get("reason") or "工具调用被运行时拦截。"),
                metadata={"guard_result": guard_result, "tool_input": tool_input},
            )

        payload = _latest_tool_result_payload(next_state, requested_tool_name)
        if requested_tool_name == FILESYSTEM_SKILL_CAPABILITY:
            recovery_outcome = _no_dead_end_outcome_from_payload(payload)
            next_state = _mark_filesystem_write_follow_up_contract(
                next_state,
                original_tool_input=tool_input,
                recovery_outcome=recovery_outcome,
                payload=payload,
            )
            state_holder["state"] = next_state
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
        if isinstance(payload, dict):
            # Preserve runtime-owned recovery signals for the next model turn.
            # The router and executor may know that a failure is recoverable,
            # but the loop controller can only continue when that fact crosses
            # the tool-observation boundary without semantic reinterpretation.
            # Preserve runtime-owned recovery signals through every result
            # envelope. The coarse filesystem entry can fail first with a
            # structured "operation required" result, then the model may
            # choose read_file; the completion guard needs both observations
            # in the same native loop to prevent an early prose answer.
            for key in ("error_code", "recoverable", "retryable", "next_action", "missing_information", "missing_args"):
                value = _first_nested_runtime_field(payload, key)
                if value is not None:
                    observation_metadata[key] = value
            recovery_outcome = _no_dead_end_outcome_from_payload(payload)
            if isinstance(recovery_outcome, dict):
                observation_metadata["recovery_outcome"] = dict(recovery_outcome)
        goal_validation = _goal_validation_from_payload(payload)
        if goal_validation is not None:
            observation_metadata["goal_validation"] = goal_validation
        if reflection is not None:
            observation_metadata["reflection"] = reflection
        retry_input = _reflection_retry_input_from_metadata(
            reflection,
            original_tool_input=tool_input,
            requested_tool_name=requested_tool_name,
            dependencies=runtime_dependencies,
        )
        # A filesystem Skill can return a concrete read-before-write recovery
        # signal while goal validation still describes the incomplete mutation
        # as "ask_user". Prefer the executable recovery route so the runtime
        # can gather the missing evidence before asking the user again.
        suggested_next_decision = _no_dead_end_recovery_decision_from_payload(requested_tool_name, tool_input, payload)
        if suggested_next_decision is None:
            suggested_next_decision = _goal_recovery_decision_from_validation(requested_tool_name, tool_input, payload)
        if suggested_next_decision is None and retry_input is not None:
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
        stage_context = _loop_runner_stage_context_from_state(state)
        loop_context = {
            "mode": "model_selected_tool",
            "candidate_selection": selection.to_metadata_dict(),
            **_tool_choice_loop_context_hints(state),
        }
        completion_contract = state.context_metadata.get(FILESYSTEM_COMPLETION_CONTRACT_METADATA_KEY)
        if isinstance(completion_contract, dict) and (
            completion_contract.get("awaiting_write")
            or completion_contract.get("requires_follow_up_after_read")
        ):
            loop_context["filesystem_completion_contract"] = dict(completion_contract)
        result = ToolChoiceLoopRunner(
            registry=capability_registry,
            llm_client=runtime_dependencies.llm_client,
            db_session=runtime_dependencies.db_session,
            execute_tool=execute_tool,
        ).run(
            LoopAgentTask(
                user_message=state.user_message,
                available_capabilities=tuple(capabilities),
                source_type="agent_chat",
                context=loop_context,
                stage_context=stage_context,
            ),
            max_steps=_tool_choice_loop_max_steps(stage_context, capabilities=capabilities),
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
        logger.exception(
            "Tool choice loop aborted due to runtime integration error: capability_override=%s state_step=%s",
            capabilities_override,
            state.current_step,
        )
        return state

    state = state_holder["state"]
    metadata = {
        **state.context_metadata,
        "tool_candidate_selection": selection_metadata,
        "tool_choice_loop": result.to_metadata_dict(),
    }
    if result.stop_reason == LoopAgentStopReason.WAITING_USER:
        pending = result.pending_decision.to_metadata_dict() if result.pending_decision else {}
        pending_decision = result.pending_decision
        preserved_waiting_step = state.current_step if state.current_step in {"wait_confirmation", "wait_user_input"} else None
        if preserved_waiting_step == "wait_confirmation":
            return state.with_updates(context_metadata=metadata)
        return state.with_updates(
            current_step="wait_user_input",
            guard_result={
                "ok": False,
                "error_code": "RUNTIME_COMPLETION_GUARD",
                "reason": str(pending_decision.reason if pending_decision else "文件任务尚未完成。"),
                "user_message": str(pending_decision.message if pending_decision else "文件任务尚未完成，请继续提交结构化文件操作。"),
                "next_action": "wait_user_input",
                "retryable": True,
                "pending_decision": pending,
            },
            response_mode="runtime_completion_guard",
            context_metadata=metadata,
        )
    if result.final_answer:
        return state.with_updates(
            final_response=result.final_answer,
            response_mode="llm_tool_choice_loop",
            context_metadata=metadata,
        )
    return state.with_updates(context_metadata=metadata)


def _first_nested_runtime_field(payload: dict[str, Any] | None, key: str) -> Any | None:
    """Find one runtime-owned field without interpreting user prose."""

    if not isinstance(payload, dict):
        return None
    queue: list[Any] = [payload]
    visited: set[int] = set()
    while queue:
        current = queue.pop(0)
        if not isinstance(current, dict) or id(current) in visited:
            continue
        visited.add(id(current))
        if key in current and current[key] is not None:
            return current[key]
        for child_key in ("result", "raw_result", "result_envelope", "error_details", "output_payload"):
            child = current.get(child_key)
            if isinstance(child, dict):
                queue.append(child)
    return None


def _loop_runner_stage_context_from_state(state: AgentState) -> dict[str, Any] | None:
    if not isinstance(state.context_metadata, dict):
        return None
    value = state.context_metadata.get(LOOP_RUNNER_STAGE_CONTEXT_METADATA_KEY)
    return dict(value) if isinstance(value, dict) and value else None


def _tool_choice_loop_max_steps(stage_context: dict[str, Any] | None, *, capabilities: tuple[str, ...] = ()) -> int:
    if not isinstance(stage_context, dict) or not stage_context:
        base_steps = 2
    else:
        stage_plan = stage_context.get("stage_plan")
        if not isinstance(stage_plan, list) or not stage_plan:
            base_steps = 2
        else:
            base_steps = max(2, min(6, len(stage_plan) + 1))
    if FILESYSTEM_SKILL_CAPABILITY in capabilities:
        # A filesystem task may need: recover missing operation -> read source
        # -> model selects name/destination -> approval. Keep this bounded but
        # large enough that a prose-only model turn cannot consume the budget.
        return max(base_steps, 6)
    return base_steps


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
        DATABASE_COMPANY_LIST_TOOL: "本地公司列表",
        EXTERNAL_WEB_SEARCH_TOOL: "网页搜索",
        LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL: "本地企业库概览",
        LOCAL_JOB_SOURCE_OVERVIEW_TOOL: "岗位来源概览",
        OFFERIO_COMPANY_JOBS_TOOL: "OfferIO 岗位同步",
        APPLICATION_FIND_APPLY_ENTRY_TOOL: "申请入口发现",
        FILESYSTEM_SKILL_CAPABILITY: "文件系统 Skill",
        "filesystem.list_dir": "查看目录",
        "filesystem.path_exists": "检查路径是否存在",
        "filesystem.path_stat": "查看文件信息",
        "filesystem.read_file": "读取文件",
        "filesystem.write_text": "写入文件",
        "filesystem.replace_text": "精确替换文本",
        "filesystem.copy_file": "复制文件",
        "filesystem.move_file": "移动/重命名文件",
        "filesystem.delete_path": "删除文件",
        "filesystem.make_dir": "创建目录",
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
    # The complete source-eligible catalog is intentionally offered to the
    # model. Keyword selection here used to make a second semantic decision
    # before the model saw the conversation history. Runtime validation still
    # decides whether a submitted call may execute.
    capability_registry = _runtime_capability_registry(dependencies)
    capabilities = model_capability_catalog_for_agent_chat(capability_registry)
    return ToolCandidateSelection(
        capabilities=capabilities,
        signals=("model_driven_capability_catalog",),
        reasons={
            capability: "registered agent_chat capability; model decides whether it is relevant"
            for capability in capabilities
        },
    )


def model_capability_catalog_for_agent_chat(registry: Any) -> tuple[str, ...]:
    """Build the main Agent's source-eligible capability directory.

    This is deliberately independent of the current user sentence. Internal
    child-agent MCP tools and legacy fine-grained filesystem tools are runtime
    implementation details, so only the high-level child capability or Skill
    is visible to the main model.
    """

    capabilities: list[str] = []
    for definition in registry.list_definitions():
        capability = str(
            getattr(definition, "capability_id", None)
            or getattr(definition, "name", "")
        ).strip()
        if not capability or capability in LEGACY_FILESYSTEM_CAPABILITIES:
            continue
        if capability.startswith(SDK_AGENT_CHILD_MCP_PREFIXES):
            continue
        if not bool(getattr(definition, "enabled", True)):
            continue
        allowed_source_types = frozenset(getattr(definition, "allowed_source_types", ()) or ())
        if allowed_source_types and "agent_chat" not in allowed_source_types:
            continue
        capabilities.append(capability)
    return tuple(sorted(set(capabilities)))


_LOCAL_FILE_REFERENCE_RE = re.compile(
    r"[A-Za-z]:[\\/][^\r\n`\"<>]*?\.(?:tex|md|txt|pdf|docx|json|csv|yaml|yml)",
    re.IGNORECASE,
)


def _tool_choice_loop_selection_text(state: AgentState) -> str:
    user_message = str(state.user_message or "")
    if _extract_local_file_references(user_message) or not _should_reuse_recent_context_for_tool_choice(user_message):
        return user_message
    recent_context = _recent_user_context_for_tool_choice(state)
    if not recent_context:
        return user_message
    return f"{user_message}\n\n上文用户消息：\n{recent_context}"


def _tool_choice_loop_context_hints(state: AgentState) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    if _should_reuse_recent_context_for_tool_choice(str(state.user_message or "")):
        recent_context = _recent_user_context_for_tool_choice(state)
        file_context_hints = build_file_context_hints(
            user_message=str(state.user_message or ""),
            recent_user_context=recent_context,
            context=state.context_metadata,
        )
        recent_paths = _extract_local_file_references(recent_context)
        if recent_context or recent_paths:
            hints.update(
                {
                    "recent_user_context": recent_context,
                    "context_usage_hint": "如果当前用户用这个、它、刚才、继续等省略说法，要先复用最近用户消息里的对象、路径和约束，再判断是否调用工具。",
                    **file_context_hints,
                }
            )
            if recent_paths:
                hints["recent_file_paths"] = recent_paths

    # When a runtime recovery step has just read or inspected a file, preserve
    # that observation in the next model decision. Without it, a fresh
    # candidate loop only sees the original user wording and can hallucinate a
    # completed rename instead of submitting a concrete, approvable target.
    tool_observations: list[str] = []
    for message in state.llm_messages or []:
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        role = str(message.get("role") or "")
        tool_status = str(metadata.get("tool_status") or "").strip()
        is_tool_observation = role == "tool" or (
            metadata.get("source") == "tool_transcript"
            and tool_status in {"succeeded", "failed", "waiting_user", "blocked"}
        )
        if is_tool_observation:
            tool_observations.append(content)
    if tool_observations:
        hints["tool_observations"] = "\n\n".join(tool_observations[-4:])[-6000:]
    if not hints:
        return {}
    return hints


def _recent_user_context_for_tool_choice(state: AgentState, *, limit: int = 4, max_chars: int = 1600) -> str:
    messages: list[str] = []
    for message in reversed(state.llm_messages or []):
        if str(message.get("role") or "") != "user":
            continue
        content = str(message.get("content") or "").strip()
        if not content or content == state.user_message:
            continue
        messages.append(content)
        if len(messages) >= limit:
            break
    context = "\n".join(reversed(messages))
    if len(context) <= max_chars:
        return context
    return context[-max_chars:]


def _recent_user_context_from_messages(
    llm_messages: list[dict[str, Any]],
    *,
    current_user_message: str,
    limit: int = 4,
    max_chars: int = 1600,
) -> str:
    messages: list[str] = []
    for message in reversed(llm_messages or []):
        if str(message.get("role") or "") != "user":
            continue
        content = str(message.get("content") or "").strip()
        if not content or content == current_user_message:
            continue
        messages.append(content)
        if len(messages) >= limit:
            break
    context = "\n".join(reversed(messages))
    if len(context) <= max_chars:
        return context
    return context[-max_chars:]


def _extract_local_file_references(text: str) -> list[str]:
    return extract_local_file_references(text)


def _dedupe_tool_choice_strings(values) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _should_reuse_recent_context_for_tool_choice(text: str) -> bool:
    return _looks_like_contextual_file_operation(text) or _looks_like_contextual_entity_tool_followup(text)


def _looks_like_contextual_entity_tool_followup(text: str) -> bool:
    if not text:
        return False
    reference_markers = ("它", "他们", "它们", "这个", "这个公司", "这个岗位", "该公司", "上面", "刚才", "继续")
    action_markers = (
        "查",
        "搜",
        "搜索",
        "看一下",
        "了解",
        "介绍",
        "主要业务",
        "主营业务",
        "是什么",
        "做什么",
        "读取",
        "打开",
        "处理",
        "修改",
        "替换",
        "换成",
        "改成",
    )
    return any(marker in text for marker in reference_markers) and any(marker in text for marker in action_markers)


def _looks_like_contextual_file_operation(text: str) -> bool:
    if not text:
        return False
    file_markers = ("文件", "文件名", "文件名称", "名字", "名称", "内容", "简历", "resume", "tex", "md", "这个", "这份", "它", "上面", "刚才")
    action_markers = (
        "读取",
        "读一下",
        "读到",
        "打开",
        "看一下",
        "看看",
        "查看",
        "处理",
        "继续",
        "修改",
        "替换",
        "换成",
        "换为",
        "换了",
        "改成",
        "改为",
        "写入",
        "保存",
        "其他不要动",
        "其他的啥都不要动",
    )
    return any(marker in text for marker in file_markers) and any(marker in text for marker in action_markers)


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


def _runtime_capability_executor_ids(dependencies: AgentGraphDependencies) -> dict[str, str]:
    executor_ids = dict(dependencies.capability_executor_ids)
    # Filesystem Skill is a main-runtime capability. Even when an SDK child
    # advertises a coarse filesystem capability, local file operations must
    # stay on the main runtime so path checks, approval, scripts and postchecks
    # all run against the real filesystem instead of a child sandbox artifact.
    executor_ids[FILESYSTEM_SKILL_CAPABILITY] = FILESYSTEM_SKILL_EXECUTOR_ID
    for executor_id, agent in dependencies.agent_executors.items():
        for capability in _declared_agent_capabilities(agent):
            executor_ids.setdefault(capability.capability_id, str(executor_id))
    return executor_ids


def _runtime_agent_executors(dependencies: AgentGraphDependencies) -> dict[str, AbilityAgent]:
    return {
        FILESYSTEM_SKILL_EXECUTOR_ID: FilesystemSkillExecutor(session_provider=lambda: dependencies.db_session),
        **dependencies.agent_executors,
    }


def _runtime_capability_registry(dependencies: AgentGraphDependencies) -> AgentCapabilityRegistry:
    declared_capabilities = [
        capability
        for agent in dependencies.agent_executors.values()
        for capability in _declared_agent_capabilities(agent)
    ]
    declared_ids = {capability.capability_id for capability in declared_capabilities}
    # MCP definitions are child-agent implementation details. Exclude them, as
    # well as generated facades, before building the main model's capability
    # registry; then register each real Agent declaration with its metadata.
    main_runtime_tools = AgentToolRegistry(
        definition
        for definition in dependencies.registry.list_definitions()
        if (
            not definition.name.startswith(SDK_AGENT_CHILD_MCP_PREFIXES)
            and definition.name not in declared_ids
            and definition.name not in LEGACY_FILESYSTEM_CAPABILITIES
        )
    )
    registry = create_default_agent_capability_registry(
        tool_registry=main_runtime_tools,
        executor_id_by_capability=_runtime_capability_executor_ids(dependencies),
    )
    for capability in declared_capabilities:
        # Filesystem remains a main-runtime Skill even when an SDK agent also
        # declares it as an internal specialist capability.
        if capability.capability_id == FILESYSTEM_SKILL_CAPABILITY:
            continue
        registry.register(capability)
    return registry


def _runtime_capability_definition(
    capability_id: str | None,
    *,
    dependencies: AgentGraphDependencies,
) -> AgentCapabilityDefinition | None:
    return _runtime_capability_registry(dependencies).get(str(capability_id or ""))


def _skill_capability_runtime_input(
    capability_id: str | None,
    tool_input: dict[str, Any],
    *,
    state: AgentState,
) -> dict[str, Any]:
    if capability_id != FILESYSTEM_SKILL_CAPABILITY:
        return dict(tool_input)
    # The model chooses both the high-level Skill and the structured filesystem
    # operation. Runtime receives those fields unchanged and only validates,
    # approves, executes, and postchecks them. Never backfill an operation from
    # the intent detector: that would reintroduce the old read-first shortcut.
    resolved = {
        "user_task": state.user_message,
        "context_metadata": dict(state.context_metadata),
        **dict(tool_input),
    }
    # Keep the public tool contract ergonomic: a model may provide a top-level
    # name_intent, while the filesystem catalog consumes it inside the
    # structured operation_intent envelope. This is a shape conversion only;
    # the runtime never invents the filename or destination semantics.
    if isinstance(resolved.get("name_intent"), dict) and not isinstance(resolved.get("operation_intent"), dict):
        resolved["operation_intent"] = {"name_intent": dict(resolved["name_intent"])}
    return resolved


def _runtime_capability_guard_payload(definition: AgentCapabilityDefinition) -> dict[str, Any]:
    return {
        "ok": True,
        "error_code": None,
        "reason": None,
        "user_message": None,
        "next_action": AgentToolNextAction.CONTINUE.value,
        "retryable": False,
        "error_details": {"capability_id": definition.capability_id, "capability_kind": definition.kind},
        "cost": {},
        "artifacts": {},
    }


def _runtime_capability_approval_payload_from_result(
    result: StandardAgentResult,
    *,
    outer_capability: str,
    outer_tool_input: dict[str, Any],
    executor_id: str,
) -> dict[str, Any] | None:
    if executor_id != FILESYSTEM_SKILL_EXECUTOR_ID:
        return None
    raw_result = _jsonable(result.raw_result)
    raw_result = raw_result if isinstance(raw_result, dict) else {}
    approval_request = raw_result.get("approval_request")
    if not result.requires_user_action or not isinstance(approval_request, dict):
        return None

    reason = str(approval_request.get("reason") or result.summary or "Runtime capability requires user approval.").strip()
    user_message = str(approval_request.get("suggested_user_message") or reason).strip()
    risk_level = str(approval_request.get("risk_level") or "high").strip() or "high"
    approval_payload = raw_result.get("approval_payload") if isinstance(raw_result.get("approval_payload"), dict) else {}
    requested_tool_name = str(approval_request.get("tool_name") or outer_capability).strip()
    # The approval card is built after the filesystem Skill has completed
    # semantic admission and chosen a concrete source, destination and
    # operation. Rehydrate those canonical arguments into the continuation
    # input so approval cannot cause a second model pass to reinterpret
    # "copy ... rename as ..." as a different operation.
    tool_input = dict(outer_tool_input)
    for key in ("operation", "src", "dst", "path", "overwrite", "operation_intent"):
        if key in approval_payload:
            tool_input[key] = approval_payload[key]

    guard_result = {
        "ok": False,
        "error_code": RUNTIME_CAPABILITY_APPROVAL_ERROR_CODE,
        "reason": reason,
        "user_message": user_message,
        "next_action": "wait_confirmation",
        "retryable": True,
        "error_details": {
            "executor_id": executor_id,
            "outer_capability": outer_capability,
            "requested_tool_name": requested_tool_name,
            "tool_input": tool_input,
            "approval_payload": dict(approval_payload),
            "risk_level": risk_level,
            "approval_type": approval_request.get("approval_type") or "runtime_capability",
        },
        "cost": {},
        "artifacts": {},
    }

    return {
        "runtime_capability_approval": True,
        "executor_id": executor_id,
        "outer_capability": outer_capability,
        "requested_tool_name": requested_tool_name,
        "tool_input": tool_input,
        "approval_payload": dict(approval_payload),
        "reason": reason,
        "user_message": user_message,
        "risk_level": risk_level,
        "approval_request": dict(approval_request),
        "capability_result": raw_result,
        "guard_result": guard_result,
    }


def _with_runtime_capability_approval_metadata(
    metadata: dict[str, Any],
    approval_payload: dict[str, Any],
) -> dict[str, Any]:
    return {**dict(metadata), RUNTIME_CAPABILITY_APPROVAL_METADATA_KEY: dict(approval_payload)}


def _runtime_capability_approval_payload_from_state_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    payload = metadata.get(RUNTIME_CAPABILITY_APPROVAL_METADATA_KEY) if isinstance(metadata, dict) else None
    return dict(payload) if isinstance(payload, dict) and payload.get("runtime_capability_approval") else None


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

    definition = _runtime_capability_definition(capability, dependencies=dependencies)
    if definition is None:
        return _blocked_execution_planner_response(
            state,
            reason=f"Planner requested an unregistered capability: {capability}",
            details={"requested_capability": capability},
        )

    tool_input = _skill_capability_runtime_input(capability, dict(action.arguments or {}), state=state)
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
    capability_registry = _runtime_capability_registry(dependencies)

    context_pack = state.context_metadata.get("context_pack") if isinstance(state.context_metadata, dict) else None
    if not isinstance(context_pack, dict):
        return state
    allowed_capabilities = [str(name) for name in context_pack.get("allowed_capabilities") or [] if str(name).strip()]
    if not allowed_capabilities:
        return state

    tool_bundle = _build_native_tool_schema_bundle(capability_registry, allowed_capabilities)
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
    tool_bundle = _build_native_tool_schema_bundle(capability_registry, allowed_capabilities)
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
                # OpenClaw-style recovery: a native tool call that was blocked
                # before execution still produces a tool result. Feed that
                # result back into the same bounded loop so the model can
                # repair its call instead of treating the block as a final answer.
                state = prepared
                blocked_payload = _latest_tool_result_payload(state, state.requested_tool_name or "")
                blocked_trace = _native_blocked_tool_trace_entry(
                    len(loop_trace) + 1,
                    state.requested_tool_name or "unknown_tool",
                    blocked_payload,
                    state,
                )
                loop_trace.append(blocked_trace)
                state = state.with_updates(
                    context_metadata=_with_loop_agent_metadata(
                        state.context_metadata,
                        {
                            **_loop_agent_progress_metadata(loop_trace),
                            "last_outcome": "blocked",
                        },
                    )
                )
                if state.current_step in {"wait_confirmation", "wait_user_input"}:
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
                continue
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
            _emit_loop_observation_events(
                dependencies,
                state=state,
                requested_tool_name=requested_tool_name,
                tool_input=tool_input,
                trace_entry=trace_entry,
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
                _emit_loop_observation_events(
                    dependencies,
                    state=state,
                    requested_tool_name=requested_tool_name,
                    tool_input=retry_input,
                    trace_entry=retry_trace_entry,
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
    tool_input = dict(getattr(tool_call, "arguments", {}) or {})
    tool_call_id = str(getattr(tool_call, "id", "") or "") or None
    if requested_tool_name not in allowed_capabilities:
        return _blocked_native_tool_call_response(
            state,
            dependencies=dependencies,
            requested_tool_name=requested_tool_name,
            tool_input=tool_input,
            tool_call_id=tool_call_id,
            reason=f"Model requested a tool outside this turn's ContextPack: {requested_tool_name}",
            details={"requested_tool_name": requested_tool_name, "allowed_capabilities": allowed_capabilities},
        )

    definition = _runtime_capability_definition(requested_tool_name, dependencies=dependencies)
    if definition is None:
        return _blocked_native_tool_call_response(
            state,
            dependencies=dependencies,
            requested_tool_name=requested_tool_name,
            tool_input=tool_input,
            tool_call_id=tool_call_id,
            reason=f"Model requested an unregistered capability: {requested_tool_name}",
            details={"requested_tool_name": requested_tool_name},
        )

    tool_input = _skill_capability_runtime_input(
        requested_tool_name,
        tool_input,
        state=state,
    )
    validation_error = _validate_native_tool_input(definition.input_schema, tool_input)
    if validation_error is not None:
        return _blocked_native_tool_call_response(
            state,
            dependencies=dependencies,
            requested_tool_name=requested_tool_name,
            tool_input=tool_input,
            tool_call_id=tool_call_id,
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


def _build_native_tool_schema_bundle(registry: Any, allowed_capabilities: list[str]) -> dict[str, Any]:
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


def _blocked_native_tool_call_response(
    state: AgentState,
    *,
    dependencies: AgentGraphDependencies,
    requested_tool_name: str,
    tool_input: dict[str, Any],
    tool_call_id: str | None,
    reason: str,
    details: dict[str, Any],
) -> AgentState:
    """Finalize a blocked native call as a model-readable tool outcome.

    Native tool calls happen before ``_maybe_tool_node``. Previously a guard or
    schema failure therefore only changed ``final_response`` and skipped the
    transcript, tool log, and ReAct observation. This helper keeps the blocked
    path aligned with normal execution: record the attempt, append a synthetic
    tool result, and let the bounded loop decide whether to retry or stop.
    """
    error_code = "NATIVE_TOOL_CALL_BLOCKED"
    next_action = "select_alternative_tool"
    if "missing required arguments" in reason.lower() or "invalid type" in reason.lower():
        error_code = "NATIVE_TOOL_INPUT_INVALID"
        next_action = "continue"
    result_payload = _tool_failure_result_payload(
        requested_tool_name,
        error=reason,
        error_code=error_code,
        retryable=True,
        next_action=next_action,
        state=state,
        definition=_runtime_capability_definition(requested_tool_name, dependencies=dependencies),
        result={
            "message": reason,
            "execution": "blocked_before_execution",
            "tool_call_id": tool_call_id,
            "details": details,
        },
    )
    tool_call = dependencies.automation_service.record_tool_call(
        ToolCallLogCreate(
            workflow_run_id=state.workflow_run_id,
            tool_name=requested_tool_name,
            tool_group="agent",
            status=ToolCallStatus.BLOCKED,
            input_payload=tool_input,
            output_payload={
                "execution": "native_tool_loop",
                "tool_call_id": tool_call_id,
                "result": result_payload,
                "details": details,
            },
            error=reason,
        )
    )
    tool_messages = _append_tool_pair_messages(
        state,
        dependencies=dependencies,
        tool_call_log_id=tool_call.id,
        tool_name=requested_tool_name,
        tool_input=tool_input,
        status="blocked",
        result=result_payload,
        error=reason,
        tool_call_id=tool_call_id,
    )
    command = AgentRunCommand(
        session_id=state.session_id,
        user_message=state.user_message,
        requested_tool_name=requested_tool_name,
        source_type="agent_chat",
        tool_input=tool_input,
    )
    _emit_runtime_tool_event(
        dependencies,
        "tool_finished",
        state=state,
        command=command,
        tool_input=tool_input,
        tool_call_id=tool_call.id,
        status="blocked",
        summary=reason,
    )
    _emit_runtime_tool_event(
        dependencies,
        "tool_result_summary",
        state=state,
        command=command,
        tool_input=tool_input,
        tool_call_id=tool_call.id,
        status="blocked",
        summary=reason,
        result_summary=_runtime_result_summary_metadata(result_payload),
    )
    logger.warning(
        "Native tool call blocked; synthetic outcome appended for ReAct recovery: tool=%s call_id=%s error_code=%s next_action=%s",
        requested_tool_name,
        tool_call_id,
        error_code,
        next_action,
    )
    return state.with_updates(
        current_step="maybe_tool",
        requested_tool_name=requested_tool_name,
        guard_result={
            "ok": False,
            "error_code": error_code,
            "reason": reason,
            "next_action": next_action,
            "retryable": True,
            "error_details": details,
        },
        tool_call_ids=[*state.tool_call_ids, tool_call.id],
        llm_messages=[*state.llm_messages, *tool_messages],
        final_response=reason,
        response_mode="llm_tool_loop",
        context_metadata=_with_tool_loop_metadata(
            state.context_metadata,
            {
                "enabled": True,
                "blocked": True,
                "last_blocked_tool_call_id": tool_call.id,
                "last_provider_tool_call_id": tool_call_id,
                "last_outcome": "blocked",
                "reason": reason,
                "details": details,
                "result": result_payload,
            },
        ),
    )


def _native_blocked_tool_trace_entry(
    iteration: int,
    requested_tool_name: str,
    payload: dict[str, Any] | None,
    state: AgentState,
) -> LoopAgentTraceEntry:
    """Expose blocked attempts in loop metadata without counting them as executed."""
    return LoopAgentTraceEntry(
        iteration=iteration,
        action=LoopAgentAction.WAIT_USER,
        capability=requested_tool_name,
        decision_reason="Runtime blocked the native call before execution and returned a synthetic tool result for recovery.",
        observation_status="blocked",
        observation_summary=_loop_agent_observation_summary(requested_tool_name, payload, state),
        tool_call_id=state.tool_call_ids[-1] if state.tool_call_ids else None,
        metadata={
            "executed": False,
            "blocked": True,
            "result_observation": _runtime_result_summary_metadata(payload),
        },
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
    metadata: dict[str, Any] = {
        "tool_input_keys": sorted(tool_input.keys()),
        "tool_input": _public_tool_input_preview(tool_input),
    }
    result_observation = _runtime_result_summary_metadata(payload)
    if result_observation:
        metadata["result_observation"] = result_observation
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
    if _state_is_waiting_for_user(state):
        return "waiting_user"
    if isinstance(payload, dict):
        goal_validation = _goal_validation_from_payload(payload)
        if isinstance(goal_validation, dict) and goal_validation.get("completed") is False and goal_validation.get("recoverable"):
            return "partial"
        return str(payload.get("status") or "unknown")
    return "unknown"


def _loop_agent_observation_summary(
    requested_tool_name: str,
    payload: dict[str, Any] | None,
    state: AgentState,
) -> str:
    if state.current_step == "wait_confirmation":
        return "Tool step paused because runtime requires user confirmation."
    if state.current_step == "wait_user_input":
        guard_result = state.guard_result if isinstance(state.guard_result, dict) else {}
        return str(guard_result.get("user_message") or guard_result.get("reason") or "工具参数还不完整，请补充后继续。")
    if not isinstance(payload, dict):
        return "Tool step completed without a structured observation payload."
    goal_validation = _goal_validation_from_payload(payload)
    if isinstance(goal_validation, dict) and goal_validation.get("completed") is False:
        reason = str(goal_validation.get("reason") or "").strip()
        if reason:
            return reason
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
    if requested_tool_name == DATABASE_COMPANY_LIST_TOOL:
        return _database_company_list_summary_response(payload)
    if requested_tool_name == LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL:
        return _company_database_overview_summary_response(payload)
    if requested_tool_name == LOCAL_JOB_SOURCE_OVERVIEW_TOOL:
        return _job_source_overview_summary_response(payload)
    if requested_tool_name == APPLICATION_FIND_APPLY_ENTRY_TOOL:
        return _apply_entry_task_summary_response(payload)
    structured_error = _structured_runtime_error_summary(payload)
    if structured_error:
        return structured_error
    error = payload.get("error") or tool_result.get("error") or result.get("error")
    if error:
        return str(error)
    return f"Tool step {payload.get('status') or 'completed'}."


def _structured_runtime_error_summary(payload: dict[str, Any]) -> str | None:
    """Expose runtime-owned recovery fields instead of collapsing them to a generic error."""

    candidates: list[dict[str, Any]] = []
    queue: list[Any] = [payload]
    visited: set[int] = set()
    while queue and len(candidates) < 12:
        current = queue.pop(0)
        if not isinstance(current, dict) or id(current) in visited:
            continue
        visited.add(id(current))
        candidates.append(current)
        for key in ("result", "raw_result", "result_envelope", "error_details"):
            child = current.get(key)
            if isinstance(child, dict):
                queue.append(child)

    for candidate in candidates:
        error_code = str(candidate.get("error_code") or candidate.get("guard_error_code") or "").strip()
        error = str(candidate.get("error") or candidate.get("reason") or "").strip()
        missing = candidate.get("missing_args") or candidate.get("missing_information") or candidate.get("missing_required_fields")
        if isinstance(missing, str):
            missing = [missing]
        missing_fields = [str(item).strip() for item in missing or [] if str(item).strip()]
        next_action = str(candidate.get("next_action") or "").strip()
        recoverable = candidate.get("recoverable") is True or candidate.get("retryable") is True
        if not (error_code or missing_fields or (error and (candidate.get("ok") is False or recoverable))):
            continue
        parts = ["runtime 结构化错误"]
        if error_code:
            parts.append(error_code)
        if error:
            parts.append(error)
        if missing_fields:
            parts.append(f"缺少字段：{', '.join(missing_fields)}")
        if next_action:
            parts.append(f"下一步：{next_action}")
        return "；".join(parts) + "。"
    return None


def _goal_recovery_decision_from_validation(
    requested_tool_name: str,
    tool_input: dict[str, Any],
    payload: dict[str, Any] | None,
) -> LoopAgentDecision | None:
    goal_validation = _goal_validation_from_payload(payload)
    if not isinstance(goal_validation, dict):
        return None
    if goal_validation.get("completed") is True or not goal_validation.get("recoverable"):
        return None
    next_action = str(goal_validation.get("next_action") or "").strip()
    reason = str(goal_validation.get("reason") or "目标校验未完成。").strip()
    if next_action == "continue_loop":
        suggested_capability = str(goal_validation.get("suggested_capability") or requested_tool_name or "").strip()
        suggested_operation = str(goal_validation.get("suggested_operation") or "").strip()
        recovery_input = {**dict(tool_input), "goal_recovery": True}
        if suggested_operation:
            recovery_input["operation"] = suggested_operation
        # No-dead-end recovery: the next loop step receives an explicit route
        # instead of only seeing a vague "tool succeeded" observation.
        return LoopAgentDecision(
            action=LoopAgentAction.CALL_TOOL,
            capability=suggested_capability or requested_tool_name,
            tool_input=recovery_input,
            reason=f"目标校验未完成：{reason}",
            metadata={"source": "goal_validation", "goal_validation": goal_validation},
        )
    if next_action in {"ask_user", "ask_user_or_search_text", "wait_user_input"}:
        return LoopAgentDecision(
            action=LoopAgentAction.WAIT_USER,
            capability=requested_tool_name,
            message=reason or "需要补充信息后继续。",
            reason=reason,
            metadata={"source": "goal_validation", "goal_validation": goal_validation},
        )
    return None


def _no_dead_end_recovery_decision_from_payload(
    requested_tool_name: str,
    tool_input: dict[str, Any],
    payload: dict[str, Any] | None,
) -> LoopAgentDecision | None:
    outcome = _no_dead_end_outcome_from_payload(payload)
    if not isinstance(outcome, dict) or outcome.get("recoverable") is not True:
        return None
    next_action = str(outcome.get("next_action") or "").strip()
    reason = str(outcome.get("reason") or outcome.get("summary") or "工具结果未完成。").strip()
    if next_action == "retry_operation":
        suggested_operation = str(outcome.get("suggested_operation") or "").strip()
        if not suggested_operation:
            return None
        recovery_input = {**dict(tool_input), "operation": suggested_operation, "no_dead_end_recovery": True}
        return LoopAgentDecision(
            action=LoopAgentAction.CALL_TOOL,
            capability=requested_tool_name,
            tool_input=recovery_input,
            reason=f"工具结果未完成，按 no-dead-end 路标继续：{reason}",
            metadata={"source": "no_dead_end", "no_dead_end_outcome": outcome},
        )
    if next_action == "read_before_write":
        read_before_write = outcome.get("read_before_write") if isinstance(outcome.get("read_before_write"), dict) else {}
        read_path = str(
            read_before_write.get("path")
            or (tool_input.get("src") if isinstance(tool_input, dict) else "")
            or (tool_input.get("path") if isinstance(tool_input, dict) else "")
            or ""
        ).strip()
        if not read_path:
            return None
        return LoopAgentDecision(
            action=LoopAgentAction.CALL_TOOL,
            capability=requested_tool_name,
            tool_input={"operation": "read_file", "path": read_path},
            reason=f"{reason}，运行时先读取文件内容，再继续选择文件名。",
            metadata={"source": "no_dead_end", "read_before_write": True, "no_dead_end_outcome": outcome},
        )
    if next_action in {"ask_user", "wait_user_input"}:
        return LoopAgentDecision(
            action=LoopAgentAction.WAIT_USER,
            capability=requested_tool_name,
            message=str(outcome.get("ask_user_message") or reason or "需要补充信息后继续。"),
            reason=reason,
            metadata={"source": "no_dead_end", "no_dead_end_outcome": outcome},
        )
    return None


def _no_dead_end_outcome_from_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    queue: list[Any] = [payload]
    visited: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if not isinstance(candidate, dict) or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if candidate.get("ok") is False and candidate.get("recoverable") is True and candidate.get("next_action"):
            return candidate
        for child_key in ("result", "raw_result", "result_envelope", "error_details", "output_payload"):
            child = candidate.get(child_key)
            if isinstance(child, dict):
                queue.append(child)
    return None


def _goal_validation_from_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    candidates: list[Any] = [payload.get("goal_validation")]
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    nested_result = result.get("result") if isinstance(result.get("result"), dict) else {}
    envelope = payload.get("result_envelope") if isinstance(payload.get("result_envelope"), dict) else None
    if envelope is None and isinstance(result.get("result_envelope"), dict):
        envelope = result.get("result_envelope")
    if envelope is None and isinstance(nested_result.get("result_envelope"), dict):
        envelope = nested_result.get("result_envelope")
    candidates.extend(
        [
            result.get("goal_validation"),
            nested_result.get("goal_validation"),
            envelope.get("goal_validation") if isinstance(envelope, dict) else None,
        ]
    )
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    return None


def _with_execution_planner_metadata(metadata: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    existing = metadata.get("execution_planner") if isinstance(metadata.get("execution_planner"), dict) else {}
    return {**metadata, "execution_planner": {**existing, **updates}}


def _with_capability_routing_metadata(metadata: dict[str, Any], decision: RouteDecision) -> dict[str, Any]:
    decision_metadata = decision.to_metadata_dict()
    return _with_runtime_decision_trace(
        {**metadata, "capability_routing": decision_metadata},
        "capability_routing",
        decision_metadata,
    )


def _with_runtime_decision_trace(metadata: dict[str, Any], stage: str, payload: dict[str, Any]) -> dict[str, Any]:
    existing = metadata.get("runtime_decision_trace") if isinstance(metadata.get("runtime_decision_trace"), list) else []
    entry = {
        "stage": stage,
        "payload": _compact_runtime_decision_payload(payload),
    }
    # Keep only the latest entries: this trace is for debugging runtime routing,
    # not for storing the full conversation or tool output in every checkpoint.
    return {**metadata, "runtime_decision_trace": [*existing, entry][-30:]}


def _compact_runtime_decision_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    allowed_keys = {
        "intent",
        "intent_reason",
        "allowed_capabilities",
        "excluded_capabilities",
        "active_file",
        "filesystem_operation",
        "operation_intent",
        "goal_state",
        "capabilities",
        "signals",
        "reasons",
        "route",
        "capability",
        "reason",
        "tool_name",
        "status",
        "tool_input",
        "error",
        "filesystem_trace",
        "goal_validation",
        "goal_completed",
        "goal_next_action",
    }
    for key, value in payload.items():
        if key not in allowed_keys or value in (None, "", [], {}):
            continue
        if key == "tool_input" and isinstance(value, dict):
            compact[key] = _public_tool_input_preview(value)
        elif key == "filesystem_trace" and isinstance(value, dict):
            compact[key] = _runtime_filesystem_trace({"filesystem_trace": value})
        else:
            compact[key] = value
    return compact


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
    if isinstance(tool_loop, dict):
        tool_input = tool_loop.get("pending_tool_input")
        if isinstance(tool_input, dict):
            return dict(tool_input)
    completion = state.context_metadata.get("tool_input_completion") if isinstance(state.context_metadata, dict) else None
    if isinstance(completion, dict):
        tool_input = completion.get("tool_input")
        if isinstance(tool_input, dict) and tool_input:
            return dict(tool_input)
    tool_choice_loop = state.context_metadata.get("tool_choice_loop") if isinstance(state.context_metadata, dict) else None
    if not isinstance(tool_choice_loop, dict):
        return {}
    pending_decision = tool_choice_loop.get("pending_decision")
    if not isinstance(pending_decision, dict):
        return {}
    tool_input = pending_decision.get("tool_input")
    return dict(tool_input) if isinstance(tool_input, dict) else {}


def _has_prepared_final_response(state: AgentState) -> bool:
    return bool(state.final_response and state.response_mode != "deterministic_stub")


def _build_context_node(state: AgentState, *, dependencies: AgentGraphDependencies) -> AgentState:
    config = dependencies.context_build_config
    if dependencies.skill_repository is not None:
        dependencies.skill_repository.ensure_builtin_content_source_skills()
    built = MemoryContextBuilder(
        dependencies.conversation_service,
        memory_repository=dependencies.memory_repository,
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
                workflow_run_id=state.workflow_run_id,
                agent_run_id=state.agent_run_id,
                target_scope=state.source_type,
            )
            memory_flush_metadata = (compact_result.summary.metadata_json or {}).get("pre_compaction_memory_flush")
            built = MemoryContextBuilder(
                dependencies.conversation_service,
                memory_repository=dependencies.memory_repository,
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
                "auto_compaction_memory_flush": memory_flush_metadata if isinstance(memory_flush_metadata, dict) else None,
            }
        except ValueError as exc:
            auto_compaction_metadata = {
                "auto_compacted": False,
                "auto_compaction_error": str(exc),
            }
    metadata = built.context_metadata
    metadata = {**metadata, **auto_compaction_metadata}
    file_context_hints = build_file_context_hints(
        user_message=state.user_message,
        recent_user_context=_recent_user_context_from_messages(built.llm_messages, current_user_message=state.user_message),
        context=metadata,
    )
    intent_frame, context_pack = _build_context_pack(
        state.user_message,
        recent_user_context=_recent_user_context_from_messages(
            built.llm_messages,
            current_user_message=state.user_message,
        ),
        registry=dependencies.registry,
        intent_detector=dependencies.intent_detector,
        capability_registry=_runtime_capability_registry(dependencies),
        source_type=state.source_type,
    )
    context_pack_metadata = context_pack.to_metadata_dict()
    metadata = {
        **metadata,
        **file_context_hints,
        "intent_frame": intent_frame.model_dump(mode="json"),
        "context_pack": context_pack_metadata,
        "context_engineering": {
            "version": "intent_context_pack_v1",
            "file_context_version": "active_file_context_v1",
            "structured_filesystem_intent_version": "v2",
            "planner_enabled": dependencies.execution_planner is not None,
        },
    }
    # GoalState is the runtime's explicit answer to "what does success mean for this turn?".
    # Tool execution can succeed while the user goal is still unfinished, so later nodes
    # validate tool output against this structured target instead of only checking ok=true.
    goal_state = build_goal_state(
        user_message=state.user_message,
        intent=str(intent_frame.intent),
        context_metadata=metadata,
    )
    metadata = {**metadata, GOAL_STATE_METADATA_KEY: goal_state.to_metadata_dict()}
    metadata = _with_runtime_decision_trace(
        metadata,
        "context_built",
        {
            "intent": intent_frame.intent,
            "intent_reason": intent_frame.reason,
            "allowed_capabilities": context_pack_metadata.get("allowed_capabilities"),
            "excluded_capabilities": context_pack_metadata.get("excluded_capabilities"),
            "active_file": metadata.get("active_file"),
            "filesystem_operation": metadata.get("filesystem_operation"),
            "operation_intent": metadata.get("operation_intent"),
            "goal_state": goal_state.to_metadata_dict(),
        },
    )
    llm_messages = _prepend_context_pack_message(built.llm_messages, context_pack_metadata)
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
    recent_user_context: str = "",
    registry: AgentToolRegistry,
    intent_detector: Any | None,
    capability_registry: AgentCapabilityRegistry | None = None,
    source_type: str = "agent_chat",
) -> tuple[IntentFrame, ContextPack]:
    # Intent detection is telemetry only. The main Agent already receives the
    # full conversation and capability directory; calling a second LLM here
    # would duplicate context tokens and could reintroduce a semantic gate.
    # Keep the deterministic frame for observability and goal metadata.
    intent_frame = HybridIntentDetector(llm_client=None).detect(
        user_message,
        recent_user_context=recent_user_context,
    )
    catalog = (
        CapabilityCatalog.from_agent_registry(capability_registry)
        if capability_registry is not None
        else CapabilityCatalog.from_registry(registry)
    )
    context_pack = ContextPackBuilder(catalog).build(intent_frame, source_type=source_type)
    return intent_frame, context_pack


def _prepend_context_pack_message(messages: list[dict[str, Any]], context_pack: ContextPack | dict[str, Any]) -> list[dict[str, Any]]:
    payload = context_pack.to_metadata_dict() if isinstance(context_pack, ContextPack) else dict(context_pack)
    content = (
        "OfferMaster capability catalog for this turn. Use the conversation history and this registered capability "
        "directory to decide whether a model-native tool call is appropriate. The runtime validates capability "
        "permission, arguments, risk and approval before execution.\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )
    return [{"role": "system", "content": content, "metadata": {"source": "context_pack"}}, *messages]


def _complete_runtime_tool_command(
    command: AgentRunCommand,
    *,
    state: AgentState,
    definition: AgentToolDefinition | AgentCapabilityDefinition | None,
) -> tuple[AgentRunCommand, dict[str, Any], ToolInputCompletionResult]:
    resolved_input = _skill_capability_runtime_input(
        command.requested_tool_name,
        _resolved_tool_input(command, state),
        state=state,
    )
    recent_user_context = _recent_user_context_for_tool_choice(state)
    file_context_hints = build_file_context_hints(
        user_message=state.user_message,
        recent_user_context=recent_user_context,
        context=state.context_metadata,
    )
    completion = complete_tool_input(
        tool_name=command.requested_tool_name or "",
        tool_input=resolved_input,
        input_schema=definition.input_schema if definition is not None else {},
        user_message=state.user_message,
        recent_user_context=recent_user_context,
        context={**state.context_metadata, **file_context_hints},
    )
    return replace(command, tool_input=completion.tool_input), completion.tool_input, completion


def _with_tool_input_completion_metadata(
    metadata: dict[str, Any],
    command: AgentRunCommand,
    completion: ToolInputCompletionResult,
) -> dict[str, Any]:
    if not completion.filled_fields and not completion.missing_required_fields:
        return {key: value for key, value in metadata.items() if key != "tool_input_completion"}
    next_metadata = {
        **metadata,
        "tool_input_completion": {
            "tool_name": command.requested_tool_name,
            "tool_input": dict(completion.tool_input),
            **completion.to_metadata_dict(),
        },
    }
    pending = build_pending_operation_from_tool_input_completion(
        tool_name=command.requested_tool_name or "",
        completion=completion,
        context_metadata=metadata,
        user_message=command.user_message,
    )
    if pending is not None:
        # tool_input_completion describes this attempt; pending_operation is the
        # resumable user task that a later follow-up can continue.
        next_metadata = with_pending_operation(next_metadata, pending)
    return next_metadata


def _tool_input_validation_guard_payload(
    command: AgentRunCommand,
    *,
    validation_error: str,
    completion: ToolInputCompletionResult,
) -> dict[str, Any]:
    missing_text = "、".join(completion.missing_required_fields)
    user_message = f"工具参数还不完整：缺少 {missing_text}。" if missing_text else "工具参数不符合要求。"
    return {
        "ok": False,
        "error_code": "TOOL_INPUT_INVALID",
        "reason": validation_error,
        "user_message": user_message,
        "next_action": "wait_user_input",
        "retryable": True,
        "error_details": {
            "requested_tool_name": command.requested_tool_name,
            "tool_input": dict(completion.tool_input),
            "missing_required_fields": list(completion.missing_required_fields),
            "completion": completion.to_metadata_dict(),
        },
        "cost": {},
        "artifacts": {},
    }


def _maybe_tool_node(
    state: AgentState,
    *,
    command: AgentRunCommand,
    dependencies: AgentGraphDependencies,
) -> AgentState:
    capability_definition = _runtime_capability_definition(command.requested_tool_name, dependencies=dependencies)
    definition = dependencies.registry.get(command.requested_tool_name or "")
    runtime_definition = capability_definition or definition
    command, tool_input, completion = _complete_runtime_tool_command(command, state=state, definition=runtime_definition)
    state = state.with_updates(context_metadata=_with_tool_input_completion_metadata(state.context_metadata, command, completion))
    validation_error = _validate_native_tool_input(runtime_definition.input_schema, tool_input) if runtime_definition is not None else None
    if validation_error is not None:
        guard_payload = _tool_input_validation_guard_payload(command, validation_error=validation_error, completion=completion)
        return state.with_updates(
            current_step="wait_user_input",
            guard_result=guard_payload,
            final_response=guard_payload["user_message"],
            response_mode="tool_input_ask_user",
        )

    # Declared Agent/Skill capabilities use their own executor-level policy.
    # The registry facade is only present so the model can select the
    # capability; applying the legacy Skill permission snapshot to that facade
    # compares `skill.filesystem` with inner names such as
    # `filesystem.copy_file` and incorrectly asks for an outer approval.
    if command.requested_tool_name != FILESYSTEM_SKILL_CAPABILITY and definition is not None and (
        capability_definition is None
        or getattr(capability_definition, "kind", "") == "tool"
    ):
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
    elif capability_definition is not None:
        # High-level Skill capabilities are not ToolRegistry entries. The
        # runtime-level capability permission check runs inside AgentRuntime.
        guard_payload = _runtime_capability_guard_payload(capability_definition)
    else:
        guard_payload = {
            "ok": False,
            "error_code": "TOOL_NOT_REGISTERED",
            "reason": f"未注册的能力：{command.requested_tool_name or ''}",
            "user_message": "模型请求了当前系统没有注册的能力。",
            "next_action": AgentToolNextAction.SELECT_ALTERNATIVE_TOOL.value,
            "retryable": False,
            "error_details": {"requested_tool_name": command.requested_tool_name or ""},
            "cost": {},
            "artifacts": {},
        }
        return state.with_updates(current_step="maybe_tool", guard_result=guard_payload)

    agent_runtime_executor_id = _agent_runtime_executor_id(command, dependencies=dependencies)
    direct_agent_registered = agent_runtime_executor_id != TOOL_REGISTRY_EXECUTOR_ID and agent_runtime_executor_id in _runtime_agent_executors(dependencies)
    input_preview = _public_tool_input_preview(tool_input)
    _emit_runtime_tool_event(
        dependencies,
        "reasoning_summary",
        state=state,
        command=command,
        tool_input=tool_input,
        status="thinking",
        summary=_runtime_reasoning_summary(command.requested_tool_name or "unknown_tool", tool_input=tool_input, user_message=state.user_message),
    )
    _emit_runtime_tool_event(
        dependencies,
        "tool_input_preview",
        state=state,
        command=command,
        tool_input=tool_input,
        status="running",
        summary=_runtime_input_preview_summary(input_preview),
        input_preview=input_preview,
    )
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
    runtime_execution_trace: dict[str, Any] | None = None
    tool_ok = False
    if capability_definition is None or dependencies.db_session is None or (definition is not None and definition.handler is None and not direct_agent_registered):
        failure_error = "Agent tool handler or database session is unavailable."
        failure_payload = _tool_failure_result_payload(
            command.requested_tool_name or "",
            error=failure_error,
            error_code="TOOL_HANDLER_UNAVAILABLE" if capability_definition is not None else "TOOL_NOT_REGISTERED",
            retryable=False,
            next_action=AgentToolNextAction.SELECT_ALTERNATIVE_TOOL.value,
            state=state,
            definition=definition,
            result={"message": failure_error, "execution": "handler_unavailable"},
        )
        failure_payload = _with_goal_validation(
            command.requested_tool_name or "",
            tool_input,
            failure_payload,
            state=state,
            registry=dependencies.registry,
        )
        tool_call = dependencies.automation_service.record_tool_call(
            ToolCallLogCreate(
                workflow_run_id=state.workflow_run_id,
                tool_name=command.requested_tool_name or "",
                tool_group="agent",
                status=ToolCallStatus.FAILED,
                input_payload=tool_input,
                output_payload={"guard_result": guard_payload, "execution": "handler_unavailable", "result": failure_payload},
                error=failure_error,
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
                "result": failure_payload,
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
            result=failure_payload,
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
        _emit_runtime_tool_event(
            dependencies,
            "tool_result_summary",
            state=state,
            command=command,
            tool_input=tool_input,
            tool_call_id=tool_call.id,
            status="failed",
            summary=tool_call.error or "工具执行失败，暂无可用结果摘要。",
            result_summary=_runtime_result_summary_metadata(failure_payload),
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
        runtime_approval_payload = _runtime_capability_approval_payload_from_result(
            agent_result,
            outer_capability=command.requested_tool_name or "",
            outer_tool_input=tool_input,
            executor_id=agent_runtime_executor_id,
        )
        if runtime_approval_payload is not None:
            guard_payload = runtime_approval_payload["guard_result"]
            _emit_runtime_tool_event(
                dependencies,
                "tool_finished",
                state=state,
                command=command,
                tool_input=tool_input,
                status="waiting_user",
                summary=str(guard_payload.get("user_message") or guard_payload.get("reason") or agent_result.summary),
            )
            return state.with_updates(
                current_step="wait_confirmation",
                guard_result=guard_payload,
                context_metadata=_with_runtime_capability_approval_metadata(state.context_metadata, runtime_approval_payload),
            )
        sdk_approval_payload = sdk_agent_approval_payload_from_result(
            agent_result,
            outer_capability=command.requested_tool_name or "",
            outer_tool_input=tool_input,
            executor_id=agent_runtime_executor_id,
        )
        if sdk_approval_payload is not None:
            guard_payload = sdk_approval_payload["guard_result"]
            result_payload = _jsonable(agent_result.raw_result)
            tool_error = f"{guard_payload.get('error_code')}: {guard_payload.get('reason')}"
            tool_call = dependencies.automation_service.record_tool_call(
                ToolCallLogCreate(
                    workflow_run_id=state.workflow_run_id,
                    tool_name=command.requested_tool_name or "",
                    tool_group="agent",
                    status=ToolCallStatus.BLOCKED,
                    input_payload=tool_input,
                    output_payload={
                        "guard_result": guard_payload,
                        "execution": "sdk_agent_approval",
                        "agent_runtime": _agent_runtime_result_metadata(agent_result, executor_id=agent_runtime_executor_id),
                        "result": result_payload,
                        "sdk_agent_approval": sdk_approval_payload,
                    },
                    error=tool_error,
                )
            )
            _mark_durable_tool_step_waiting_user(
                durable_step_id,
                state=state,
                command=command,
                tool_input=tool_input,
                dependencies=dependencies,
                tool_call_log_id=tool_call.id,
                output_payload={
                    "guard_result": guard_payload,
                    "execution": "sdk_agent_approval",
                    "agent_runtime": _agent_runtime_result_metadata(agent_result, executor_id=agent_runtime_executor_id),
                    "result": result_payload,
                    "error": tool_error,
                    "sdk_agent_approval": sdk_approval_payload,
                },
            )
            tool_messages = _append_tool_pair_messages(
                state,
                dependencies=dependencies,
                tool_call_log_id=tool_call.id,
                tool_name=command.requested_tool_name or "",
                tool_input=tool_input,
                status="waiting_user",
                result=result_payload,
                error=tool_error,
            )
            _emit_runtime_tool_event(
                dependencies,
                "tool_finished",
                state=state,
                command=command,
                tool_input=tool_input,
                tool_call_id=tool_call.id,
                status="waiting_user",
                summary=str(guard_payload.get("user_message") or guard_payload.get("reason") or agent_result.summary),
            )
            return state.with_updates(
                current_step="wait_confirmation",
                guard_result=guard_payload,
                tool_call_ids=[*state.tool_call_ids, tool_call.id],
                llm_messages=[*state.llm_messages, *tool_messages],
                context_metadata=with_sdk_agent_approval_metadata(state.context_metadata, sdk_approval_payload),
            )
        result_payload = _jsonable(agent_result.raw_result)
        if not result_payload and agent_result.status == "failed":
            result_payload = {"tool_name": command.requested_tool_name or "", "ok": False, "error": agent_result.summary}
        missing_input_state = _maybe_wait_for_missing_runtime_tool_input(
            state,
            command=command,
            tool_input=tool_input,
            agent_result=agent_result,
            result_payload=result_payload,
            guard_payload=guard_payload,
            durable_step_id=durable_step_id,
            dependencies=dependencies,
        )
        if missing_input_state is not None:
            return missing_input_state
        result_payload = _with_result_envelope(command.requested_tool_name or "", result_payload, state=state)
        result_payload = _with_goal_validation(
            command.requested_tool_name or "",
            tool_input,
            result_payload,
            state=state,
            registry=dependencies.registry,
        )
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
            event="tool_succeeded" if tool_ok else "tool_failed",
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
        result_summary = _runtime_result_summary_metadata(result_payload)
        runtime_execution_trace = _runtime_tool_execution_trace(
            command=command,
            tool_input=tool_input,
            status="succeeded" if tool_ok else "failed",
            result_payload=result_payload,
            result_summary=result_summary,
            error=tool_error,
        )
        _emit_runtime_tool_event(
            dependencies,
            "tool_result_summary",
            state=state,
            command=command,
            tool_input=tool_input,
            tool_call_id=tool_call.id,
            status="succeeded" if tool_ok else "failed",
            summary=_runtime_result_summary_text(result_summary),
            result_summary=result_summary,
        )
    except Exception as exc:  # pragma: no cover - exercised through workflow-level error tests later.
        failure_payload = _tool_failure_result_payload(
            command.requested_tool_name or "",
            error=str(exc),
            error_code="TOOL_EXECUTION_EXCEPTION",
            retryable=False,
            next_action=AgentToolNextAction.SELECT_ALTERNATIVE_TOOL.value,
            state=state,
            definition=definition,
            result={"message": str(exc), "error_type": exc.__class__.__name__},
        )
        failure_payload = _with_goal_validation(
            command.requested_tool_name or "",
            tool_input,
            failure_payload,
            state=state,
            registry=dependencies.registry,
        )
        failure_result_summary = _runtime_result_summary_metadata(failure_payload)
        runtime_execution_trace = _runtime_tool_execution_trace(
            command=command,
            tool_input=tool_input,
            status="failed",
            result_payload=failure_payload,
            result_summary=failure_result_summary,
            error=str(exc),
        )
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
                    "result": failure_payload,
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
                "result": failure_payload,
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
            result=failure_payload,
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
        _emit_runtime_tool_event(
            dependencies,
            "tool_result_summary",
            state=state,
            command=command,
            tool_input=tool_input,
            tool_call_id=tool_call.id,
            status="failed",
            summary=str(exc) or "工具执行失败，暂无可用结果摘要。",
            result_summary=failure_result_summary,
        )
    context_metadata = state.context_metadata
    if runtime_execution_trace is not None:
        context_metadata = _with_runtime_decision_trace(context_metadata, "tool_execution", runtime_execution_trace)
    if tool_ok:
        context_metadata = clear_pending_operation(context_metadata)
        context_metadata = promote_contract_resource_effects_context(
            context_metadata,
            result_payload,
            tool_input=tool_input,
            semantic_profile=getattr(runtime_definition, "semantic_profile", None),
        )
        context_metadata = promote_declared_resource_effects_context(context_metadata, result_payload)
        context_metadata = promote_filesystem_result_context(context_metadata, result_payload)
    return state.with_updates(
        current_step="maybe_tool",
        guard_result=guard_payload,
        tool_call_ids=[*state.tool_call_ids, tool_call.id],
        llm_messages=[*state.llm_messages, *tool_messages],
        context_metadata=context_metadata,
    )


def _maybe_wait_for_missing_runtime_tool_input(
    state: AgentState,
    *,
    command: AgentRunCommand,
    tool_input: dict[str, Any],
    agent_result: StandardAgentResult,
    result_payload: dict[str, Any],
    guard_payload: dict[str, Any],
    durable_step_id: str | None,
    dependencies: AgentGraphDependencies,
) -> AgentState | None:
    missing = _missing_runtime_tool_input_fields(agent_result=agent_result, result_payload=result_payload)
    if not missing:
        return None
    operation_args = result_payload.get("arguments") if isinstance(result_payload.get("arguments"), dict) else tool_input
    completion = ToolInputCompletionResult(
        tool_input=dict(operation_args),
        missing_required_fields=tuple(missing),
    )
    pending = build_pending_operation_from_tool_input_completion(
        tool_name=command.requested_tool_name or "",
        completion=completion,
        context_metadata=state.context_metadata,
        user_message=state.user_message,
    )
    if pending is None:
        return None

    logger.info(
        "Runtime tool input incomplete; persisted pending operation: capability=%s operation=%s missing=%s known_keys=%s",
        pending.capability,
        pending.operation,
        list(pending.missing_args),
        sorted(pending.known_args.keys()),
    )

    user_message = f"工具参数还不完整：缺少 {'、'.join(missing)}。"
    wait_payload = {
        "ok": False,
        "error_code": "TOOL_INPUT_INVALID",
        "reason": str(result_payload.get("error") or result_payload.get("reason") or user_message),
        "user_message": user_message,
        "next_action": "wait_user_input",
        "retryable": True,
        "error_details": {
            "requested_tool_name": command.requested_tool_name,
            "tool_input": dict(operation_args),
            "missing_required_fields": list(missing),
            "pending_operation": pending.to_metadata_dict(),
        },
        "cost": {},
        "artifacts": {},
    }
    tool_call = dependencies.automation_service.record_tool_call(
        ToolCallLogCreate(
            workflow_run_id=state.workflow_run_id,
            tool_name=command.requested_tool_name or "",
            tool_group="agent",
            status=ToolCallStatus.BLOCKED,
            input_payload=tool_input,
            output_payload={"guard_result": wait_payload, "execution": "missing_tool_input", "result": result_payload},
            error=user_message,
        )
    )
    _mark_durable_tool_step_waiting_user(
        durable_step_id,
        state=state,
        command=command,
        tool_input=tool_input,
        dependencies=dependencies,
        tool_call_log_id=tool_call.id,
        output_payload={"guard_result": wait_payload, "execution": "missing_tool_input", "result": result_payload, "error": user_message},
    )
    tool_messages = _append_tool_pair_messages(
        state,
        dependencies=dependencies,
        tool_call_log_id=tool_call.id,
        tool_name=command.requested_tool_name or "",
        tool_input=tool_input,
        status="waiting_user",
        result=result_payload,
        error=user_message,
    )
    _emit_runtime_tool_event(
        dependencies,
        "tool_finished",
        state=state,
        command=command,
        tool_input=tool_input,
        tool_call_id=tool_call.id,
        status="waiting_user",
        summary=user_message,
    )
    return state.with_updates(
        current_step="wait_user_input",
        guard_result=wait_payload,
        tool_call_ids=[*state.tool_call_ids, tool_call.id],
        llm_messages=[*state.llm_messages, *tool_messages],
        context_metadata=with_pending_operation(state.context_metadata, pending),
        final_response=user_message,
        response_mode="tool_input_ask_user",
    )


def _missing_runtime_tool_input_fields(*, agent_result: StandardAgentResult, result_payload: dict[str, Any]) -> tuple[str, ...]:
    # Only an explicit runtime argument error can create a pending tool-input
    # frame. High-level semantic gaps such as ``filesystem_operation`` mean
    # "the action is unclear", not "the tool schema is missing a field".
    # Treating generic missing_information as a schema error leaks internal
    # names to users and overwrites the Skill's natural clarification message.
    error_code = str(result_payload.get("error_code") or "").strip()
    if error_code != "MISSING_REQUIRED_ARGUMENT":
        semantic_missing = result_payload.get("missing_information") or agent_result.missing_information
        if semantic_missing:
            logger.info(
                "Runtime kept semantic gap out of pending tool-input state: fields=%s error_code=%s",
                list(semantic_missing) if isinstance(semantic_missing, (list, tuple)) else semantic_missing,
                error_code or "none",
            )
        return ()

    missing = result_payload.get("missing_args")
    if not isinstance(missing, (list, tuple)):
        return ()
    if str(result_payload.get("next_action") or "") not in {"wait_user_input", "ask_user", "ask_user_or_complete_from_pending_operation", ""}:
        return ()
    return tuple(str(field) for field in missing if str(field or "").strip())


def _run_agent_tool_through_runtime(
    command: AgentRunCommand,
    *,
    state: AgentState,
    dependencies: AgentGraphDependencies,
    tool_input: dict[str, Any],
) -> StandardAgentResult:
    capability_name = command.requested_tool_name or "unknown_agent"
    executor_id = _agent_runtime_executor_id(command, dependencies=dependencies)
    delegation_id = f"delegation:{state.workflow_run_id}:{uuid4().hex}"
    executors: dict[str, AbilityAgent] = {
        TOOL_REGISTRY_EXECUTOR_ID: ToolRegistryAgentExecutor(
            dependencies.registry,
            session_provider=lambda _context: dependencies.db_session,
        ),
        **_runtime_agent_executors(dependencies),
    }
    runtime = AgentRuntime(
        registry=_runtime_capability_registry(dependencies),
        executors=executors,
    )
    context = AgentRuntimeContext(
        session_id=state.session_id,
        run_id=state.workflow_run_id,
        task_id=f"{state.workflow_run_id}:tool-{len(state.tool_call_ids) + 1}",
        permission_scope={"source_type": command.source_type, "user_confirmed": command.user_confirmed},
        metadata={"agent_run_id": state.agent_run_id, "delegation_id": delegation_id},
        capability_id=command.requested_tool_name,
        event_sink=dependencies.event_sink,
    )
    _emit_subagent_lifecycle_event(
        dependencies,
        "subagent_started",
        state=state,
        command=command,
        tool_input=tool_input,
        executor_id=executor_id,
        delegation_id=delegation_id,
        status="running",
        summary=f"已委派 {capability_name}，子 Agent 开始执行。",
    )
    logger.info(
        "Child agent delegation started: capability=%s delegation_id=%s workflow_run_id=%s",
        capability_name,
        delegation_id,
        state.workflow_run_id,
    )
    try:
        result = runtime.call(
            AgentTask(
                capability_id=capability_name,
                goal=state.user_message,
                input_payload=tool_input,
            ),
            context,
        )
    except Exception as exc:
        _emit_subagent_lifecycle_event(
            dependencies,
            "subagent_finished",
            state=state,
            command=command,
            tool_input=tool_input,
            executor_id=executor_id,
            delegation_id=delegation_id,
            status="failed",
            summary=f"子 Agent 执行失败：{type(exc).__name__}。",
        )
        logger.exception(
            "Child agent delegation failed: capability=%s delegation_id=%s workflow_run_id=%s",
            capability_name,
            delegation_id,
            state.workflow_run_id,
        )
        raise

    result_status = str(result.status or "").strip().lower()
    lifecycle_status = "waiting_user" if result_status in {"needs_approval", "waiting_user"} or result.requires_user_action else (
        "failed" if result_status in {"failed", "error"} else "succeeded"
    )
    _emit_subagent_lifecycle_event(
        dependencies,
        "subagent_finished",
        state=state,
        command=command,
        tool_input=tool_input,
        executor_id=executor_id,
        delegation_id=delegation_id,
        status=lifecycle_status,
        summary=result.summary or f"子 Agent {lifecycle_status}。",
    )
    logger.info(
        "Child agent delegation finished: capability=%s delegation_id=%s status=%s",
        capability_name,
        delegation_id,
        lifecycle_status,
    )
    return result


def _emit_subagent_lifecycle_event(
    dependencies: AgentGraphDependencies,
    event_type: str,
    *,
    state: AgentState,
    command: AgentRunCommand,
    tool_input: dict[str, Any],
    executor_id: str,
    delegation_id: str,
    status: str,
    summary: str,
) -> None:
    """Emit one durable UI boundary for a delegated Agent run.

    Nested MCP events describe work performed inside the child Agent. This
    boundary describes the child Agent itself so clients can distinguish
    "delegated" from "registered" and correlate all nested calls reliably.
    """
    if dependencies.event_sink is None:
        return
    capability = command.requested_tool_name or "unknown_agent"
    definition = _runtime_capability_definition(capability, dependencies=dependencies)
    if definition is None or getattr(definition, "kind", None) != "agent":
        return
    payload: dict[str, Any] = {
        "event_type": event_type,
        "event_label": "子 Agent 已委派" if event_type == "subagent_started" else "子 Agent 已返回",
        "session_id": state.session_id,
        "workflow_run_id": state.workflow_run_id,
        "agent_run_id": state.agent_run_id,
        "delegation_id": delegation_id,
        "step_index": len(state.tool_call_ids) + 1,
        "tool_name": capability,
        "capability": capability,
        "capability_kind": "agent",
        "executor_id": executor_id,
        "status": status,
        "summary": summary,
        "tool_input_keys": sorted(str(key) for key in tool_input),
        "parent_capability": capability,
        "subagent_name": getattr(definition, "name", None) if definition is not None else None,
    }
    try:
        dependencies.event_sink(payload)
    except Exception:
        logger.exception(
            "Failed to emit child-agent lifecycle event: event_type=%s capability=%s delegation_id=%s",
            event_type,
            capability,
            delegation_id,
        )


def _agent_runtime_executor_id(command: AgentRunCommand, *, dependencies: AgentGraphDependencies) -> str:
    capability = command.requested_tool_name or ""
    definition = _runtime_capability_definition(capability, dependencies=dependencies)
    if definition is not None:
        return definition.executor_id
    return _runtime_capability_executor_ids(dependencies).get(capability, TOOL_REGISTRY_EXECUTOR_ID)


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
    input_preview: dict[str, Any] | None = None,
    result_summary: dict[str, Any] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    step_index: int | None = None,
) -> None:
    if dependencies.event_sink is None:
        return
    tool_name = command.requested_tool_name or "unknown_tool"
    capability_definition = _runtime_capability_definition(tool_name, dependencies=dependencies)
    payload: dict[str, Any] = {
        "event_type": event_type,
        "event_label": _runtime_tool_event_label(event_type),
        "session_id": state.session_id,
        "workflow_run_id": state.workflow_run_id,
        "agent_run_id": state.agent_run_id,
        "step_index": step_index or len(state.tool_call_ids) + 1,
        "tool_name": tool_name,
        "capability_kind": capability_definition.kind if capability_definition is not None else None,
        # Let clients render the actual executor path instead of guessing from the tool name.
        "executor_id": _agent_runtime_executor_id(command, dependencies=dependencies),
        "tool_call_id": tool_call_id,
        "status": status,
        "summary": summary,
        "tool_input_keys": sorted(str(key) for key in tool_input.keys()),
    }
    if reflection is not None:
        payload["reflection"] = dict(reflection)
    if suggested_input_patch is not None:
        payload["suggested_input_patch"] = dict(suggested_input_patch)
    if input_preview is not None:
        payload["input_preview"] = dict(input_preview)
    if result_summary is not None:
        payload["result_summary"] = dict(result_summary)
    if evidence is not None:
        payload["evidence"] = [dict(item) for item in evidence]
    dependencies.event_sink(payload)


def _runtime_tool_event_label(event_type: str) -> str:
    return {
        "reasoning_summary": "思考摘要",
        "tool_input_preview": "工具输入",
        "tool_started": "工具开始",
        "tool_finished": "工具完成",
        "tool_result_summary": "结果摘要",
        "reflection_evaluation": "反思判断",
        "tool_reflection_retry": "准备重试",
        "evidence_selected": "证据选择",
        "textual_tool_call_recovered": "自动纠偏执行",
    }.get(event_type, event_type)


def _runtime_reasoning_summary(tool_name: str, *, tool_input: dict[str, Any], user_message: str) -> str:
    if tool_name == EXTERNAL_WEB_SEARCH_TOOL:
        query = str(tool_input.get("query") or user_message).strip()
        return f"当前问题需要公开信息核对，主 agent 准备通过网页搜索确认：{query}。"
    if tool_name == DATABASE_COMPANY_LIST_TOOL:
        limit = tool_input.get("limit")
        limit_text = f"，本次最多展示 {limit} 家" if isinstance(limit, int) else ""
        return f"当前问题需要读取本地公司列表，主 agent 会使用本地只读数据库工具{limit_text}。"
    if tool_name == LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL:
        return "当前问题可以先查本地企业库，主 agent 会优先使用本地只读工具。"
    if tool_name == LOCAL_JOB_SOURCE_OVERVIEW_TOOL:
        return "当前问题涉及岗位来源概览，主 agent 会读取本地岗位来源统计。"
    if tool_name == FILESYSTEM_SKILL_CAPABILITY:
        return "当前问题属于本地文件操作，主 agent 只选择文件系统 Skill；具体读、查存在、改名、替换或复制由 Skill 内部判断。"
    return f"模型选择了能力 {tool_name}，运行时会先校验权限和输入再执行。"


def _public_tool_input_preview(tool_input: dict[str, Any]) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    for key, value in tool_input.items():
        if isinstance(value, str):
            preview[str(key)] = value[:240]
        elif isinstance(value, (int, float, bool)) or value is None:
            preview[str(key)] = value
    return preview


def _runtime_input_preview_summary(input_preview: dict[str, Any]) -> str:
    query = str(input_preview.get("query") or "").strip()
    if query:
        return f"准备使用关键词：{query}。"
    keys = "、".join(str(key) for key in input_preview.keys())
    return f"准备使用工具输入字段：{keys}。" if keys else "准备调用工具。"


def _runtime_result_summary_metadata(result_payload: Any) -> dict[str, Any]:
    if not isinstance(result_payload, dict):
        return {}
    nested = result_payload.get("result") if isinstance(result_payload.get("result"), dict) else {}
    deeper = nested.get("result") if isinstance(nested.get("result"), dict) else {}
    envelope = result_payload.get("result_envelope") if isinstance(result_payload.get("result_envelope"), dict) else None
    if envelope is None and isinstance(nested.get("result_envelope"), dict):
        envelope = nested.get("result_envelope")
    if envelope is None and isinstance(deeper.get("result_envelope"), dict):
        envelope = deeper.get("result_envelope")
    goal_validation = result_payload.get("goal_validation") if isinstance(result_payload.get("goal_validation"), dict) else None
    if goal_validation is None and isinstance(envelope, dict) and isinstance(envelope.get("goal_validation"), dict):
        goal_validation = envelope.get("goal_validation")
    filesystem_trace = _runtime_filesystem_trace(result_payload, nested, deeper)

    result_items = _first_list(deeper.get("results"), nested.get("results"), result_payload.get("results"))
    source_items = _first_list(deeper.get("sources"), nested.get("sources"), result_payload.get("sources"))
    artifact_items = _first_list(
        envelope.get("artifacts") if isinstance(envelope, dict) else None,
        deeper.get("artifacts"),
        nested.get("artifacts"),
        result_payload.get("artifacts"),
    )
    evidence = _runtime_evidence_items(artifact_items or source_items)
    domains = _runtime_source_domains(evidence, source_items)
    summary: dict[str, Any] = {}
    if result_items:
        summary["result_count"] = len(result_items)
    elif source_items:
        summary["result_count"] = len(source_items)
    if evidence:
        summary["source_count"] = len(evidence)
    elif source_items:
        summary["source_count"] = len(source_items)
    if domains:
        summary["source_domains"] = domains[:8]
    if isinstance(goal_validation, dict):
        # Keep the no-dead-end signal in the compact event summary so callers
        # can see whether the next step should continue, ask, or finish.
        summary["goal_validation"] = goal_validation
        summary["goal_completed"] = bool(goal_validation.get("completed"))
        summary["goal_next_action"] = goal_validation.get("next_action")
    if filesystem_trace is not None:
        summary["filesystem_trace"] = filesystem_trace
    return summary


def _runtime_filesystem_trace(*payloads: Any) -> dict[str, Any] | None:
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        trace = payload.get("filesystem_trace")
        if not isinstance(trace, dict):
            continue
        postcheck = trace.get("postcheck") if isinstance(trace.get("postcheck"), dict) else {}
        return {
            "operation": trace.get("operation"),
            "precheck": _compact_filesystem_trace_section(trace.get("precheck")),
            "script": _compact_filesystem_trace_section(trace.get("script")),
            "postcheck": _compact_filesystem_trace_section(postcheck),
        }
    return None


def _compact_filesystem_trace_section(section: Any) -> dict[str, Any]:
    if not isinstance(section, dict):
        return {}
    allowed = {
        "completed",
        "reason",
        "source_path",
        "target_path",
        "path",
        "source_exists_before",
        "target_exists_before",
        "source_exists_after",
        "target_exists_after",
        "path_exists_after",
        "internal_tool",
        "internal_script",
        "ok",
        "error",
        "return_code",
    }
    # Keep runtime events compact: enough to debug routing/execution, without
    # copying file contents or long subprocess output into every event.
    return {key: value for key, value in section.items() if key in allowed}


def _first_list(*values: Any) -> list[Any]:
    for value in values:
        if isinstance(value, list):
            return value
    return []


def _runtime_evidence_items(items: list[Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            url = item.strip()
            title = "证据来源"
        elif isinstance(item, dict):
            url = str(item.get("url") or item.get("href") or "").strip()
            title = str(item.get("title") or item.get("name") or item.get("source") or "证据来源").strip()
        else:
            continue
        if not url and not title:
            continue
        payload: dict[str, Any] = {"title": title or "证据来源"}
        if url:
            payload["url"] = url
        evidence.append(payload)
    return evidence


def _runtime_source_domains(evidence: list[dict[str, Any]], source_items: list[Any]) -> list[str]:
    domains: list[str] = []
    for item in evidence:
        url = str(item.get("url") or "").strip()
        domain = urlparse(url).netloc.lower().removeprefix("www.") if url else ""
        if domain and domain not in domains:
            domains.append(domain)
    for item in source_items:
        if isinstance(item, dict):
            domain = str(item.get("domain") or item.get("source_domain") or "").strip().lower()
            if domain and domain not in domains:
                domains.append(domain)
    return domains


def _runtime_result_summary_text(result_summary: dict[str, Any]) -> str:
    filesystem_trace = result_summary.get("filesystem_trace") if isinstance(result_summary.get("filesystem_trace"), dict) else None
    if filesystem_trace is not None:
        text = _filesystem_trace_summary_text(filesystem_trace)
        if text:
            return text
    parts: list[str] = []
    result_count = result_summary.get("result_count")
    source_count = result_summary.get("source_count")
    domains = result_summary.get("source_domains") if isinstance(result_summary.get("source_domains"), list) else []
    if isinstance(result_count, int):
        parts.append(f"找到 {result_count} 条结果")
    if isinstance(source_count, int):
        parts.append(f"整理出 {source_count} 条来源")
    if domains:
        parts.append("来源包括 " + "、".join(str(domain) for domain in domains[:3]))
    return "，".join(parts) + "。" if parts else "工具结果已完成，暂无可展示的来源摘要。"


def _filesystem_trace_summary_text(filesystem_trace: dict[str, Any]) -> str:
    operation = str(filesystem_trace.get("operation") or "文件操作").strip()
    postcheck = filesystem_trace.get("postcheck") if isinstance(filesystem_trace.get("postcheck"), dict) else {}
    source = str(postcheck.get("source_path") or "").strip()
    target = str(postcheck.get("target_path") or "").strip()
    path_text = f"{source} -> {target}" if source and target else str(postcheck.get("path") or "").strip()
    reason = str(postcheck.get("reason") or "").strip()
    if postcheck.get("completed") is False:
        suffix = f"，原因：{reason}" if reason else ""
        return f"文件动作复核失败：{operation}，{path_text}{suffix}。" if path_text else f"文件动作复核失败：{operation}{suffix}。"
    if postcheck.get("completed") is True:
        return f"文件动作已执行并复核：{operation}，{path_text}。" if path_text else f"文件动作已执行并复核：{operation}。"
    return ""


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
        step_index=trace_entry.iteration,
    )


def _emit_loop_observation_events(
    dependencies: AgentGraphDependencies,
    *,
    state: AgentState,
    requested_tool_name: str,
    tool_input: dict[str, Any],
    trace_entry: LoopAgentTraceEntry,
) -> None:
    metadata = trace_entry.metadata if isinstance(trace_entry.metadata, dict) else {}
    reflection = metadata.get("reflection") if isinstance(metadata.get("reflection"), dict) else None
    if isinstance(reflection, dict):
        _emit_runtime_tool_event(
            dependencies,
            "reflection_evaluation",
            state=state,
            command=AgentRunCommand(
                session_id=state.session_id,
                user_message=state.user_message,
                requested_tool_name=requested_tool_name,
                source_type="agent_chat",
                tool_input=tool_input,
            ),
            tool_input=tool_input,
            tool_call_id=trace_entry.tool_call_id,
            status=str(reflection.get("next_action") or "observed"),
            summary=str(reflection.get("reason") or "主 agent 已评估这次工具结果。"),
            reflection=reflection,
            step_index=trace_entry.iteration,
        )
    result_summary = metadata.get("result_observation") if isinstance(metadata.get("result_observation"), dict) else None
    evidence = _runtime_evidence_from_result_summary(result_summary)
    if evidence:
        _emit_runtime_tool_event(
            dependencies,
            "evidence_selected",
            state=state,
            command=AgentRunCommand(
                session_id=state.session_id,
                user_message=state.user_message,
                requested_tool_name=requested_tool_name,
                source_type="agent_chat",
                tool_input=tool_input,
            ),
            tool_input=tool_input,
            tool_call_id=trace_entry.tool_call_id,
            status="succeeded",
            summary=_runtime_evidence_summary(evidence),
            evidence=evidence,
            step_index=trace_entry.iteration,
        )


def _runtime_evidence_from_result_summary(result_summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    evidence = result_summary.get("evidence") if isinstance(result_summary, dict) else None
    if not isinstance(evidence, list):
        return []
    return [dict(item) for item in evidence if isinstance(item, dict)]


def _runtime_evidence_summary(evidence: list[dict[str, Any]]) -> str:
    titles = [str(item.get("title") or "证据来源").strip() for item in evidence[:3]]
    titles = [title for title in titles if title]
    return f"已选择 {len(evidence)} 条证据：" + "、".join(titles) + "。" if titles else f"已选择 {len(evidence)} 条证据。"


def _agent_runtime_result_metadata(result: StandardAgentResult, *, executor_id: str) -> dict[str, Any]:
    metadata = {
        "executor_id": executor_id,
        "status": result.status,
        "summary": result.summary,
        "requires_user_action": result.requires_user_action,
    }
    telemetry = _agent_runtime_compact_telemetry(result.raw_result)
    if telemetry is not None:
        metadata["telemetry"] = telemetry
    return metadata


def _agent_runtime_compact_telemetry(raw_result: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw_result, dict):
        return None
    raw_metadata = raw_result.get("metadata") if isinstance(raw_result.get("metadata"), dict) else None
    telemetry = raw_metadata.get("telemetry") if isinstance(raw_metadata, dict) and isinstance(raw_metadata.get("telemetry"), dict) else None
    if telemetry is None:
        return None
    allowed_keys = {
        "schema_version",
        "subagent_name",
        "capability_id",
        "status",
        "allowed_tool_count",
        "tool_call_count",
        "retry_count",
        "approval_required",
        "operation_ref_count",
        "sandbox",
        "delegation",
    }
    # Runtime logs need enough counters for dashboards, but should not duplicate raw child-agent traces.
    return {key: value for key, value in telemetry.items() if key in allowed_keys}


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


def _mark_durable_tool_step_waiting_user(
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
        service.mark_step_waiting_user(
            step_id,
            tool_call_log_id=tool_call_log_id,
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
    for memory_id in state.loaded_memory_ids:
        refs.append(
            {
                "memory_id": str(memory_id),
                "source_type": "agent_memory",
                "usage_reason": "ContextBuilder loaded long-term memory for this run",
                "visibility_scope": AgentMemoryVisibilityScope.MAIN_AGENT_ONLY,
                "memory_payload": {"memory_id": str(memory_id)},
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


def _runtime_tool_execution_trace(
    *,
    command: AgentRunCommand,
    tool_input: dict[str, Any],
    status: str,
    result_payload: Any,
    result_summary: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "tool_name": command.requested_tool_name,
        "status": status,
        "tool_input": dict(tool_input),
        "error": error,
    }
    if isinstance(result_summary, dict):
        for key in ("goal_validation", "goal_completed", "goal_next_action", "filesystem_trace"):
            if key in result_summary:
                trace[key] = result_summary[key]
    if "filesystem_trace" not in trace:
        filesystem_trace = _runtime_filesystem_trace(result_payload)
        if filesystem_trace is not None:
            trace["filesystem_trace"] = filesystem_trace
    return trace


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
    if command.requested_tool_name == DATABASE_COMPANY_LIST_TOOL:
        return {"limit": requested_sample_limit_from_text(state.user_message)}
    if command.requested_tool_name == LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL:
        return {"sample_limit": requested_sample_limit_from_text(state.user_message)}
    if command.requested_tool_name == LOCAL_JOB_SOURCE_OVERVIEW_TOOL:
        context_pack = state.context_metadata.get("context_pack") if isinstance(state.context_metadata, dict) else None
        context_pack = context_pack if isinstance(context_pack, dict) else {}
        sync_policy = context_pack.get("sync_policy") if isinstance(context_pack.get("sync_policy"), dict) else {}
        if sync_policy.get("mode") == "company_board_count":
            return {"mode": "company_board_count"}
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
    tool_call_id: str | None = None,
) -> list[dict[str, Any]]:
    # Keep the provider call id in the transcript when one exists. Normal
    # runtime calls use the database log id; native calls also have a model-side
    # id that is needed to diagnose blocked or unstarted requests.
    call_content_json = {"tool_name": tool_name, "input": tool_input}
    if tool_call_id:
        call_content_json["tool_call_id"] = tool_call_id
    call_message = dependencies.conversation_service.append_message(
        state.session_id,
        AgentMessageCreate(
            role=AgentMessageRole.TOOL_CALL,
            content_text=f"Tool call: {tool_name}",
            content_json=call_content_json,
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
            content_json={
                "tool_name": tool_name,
                "status": status,
                "result": result,
                "error": error,
                **({"tool_call_id": tool_call_id} if tool_call_id else {}),
            },
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


def _tool_failure_result_payload(
    tool_name: str,
    *,
    error: str,
    error_code: str,
    retryable: bool,
    next_action: str,
    state: AgentState,
    definition: AgentToolDefinition | None,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tool_name": tool_name,
        "ok": False,
        "error": error,
        "error_code": error_code,
        "retryable": retryable,
        "next_action": next_action,
        "result": dict(result or {}),
    }
    envelope = build_tool_result_envelope(
        capability=tool_name,
        status="failed",
        executor=TOOL_REGISTRY_EXECUTOR_ID,
        risk_level=_tool_result_risk_level(state=state, definition=definition),
        result_payload=payload,
        source_type=state.source_type,
    )
    return {**payload, "result_envelope": envelope.to_dict()}


def _tool_result_risk_level(*, state: AgentState, definition: AgentToolDefinition | None) -> str:
    if definition is not None:
        risk_level = getattr(definition, "risk_level", None)
        risk_value = getattr(risk_level, "value", risk_level)
        if risk_value:
            return str(risk_value)
    context_pack = state.context_metadata.get("context_pack") if isinstance(state.context_metadata, dict) else None
    if isinstance(context_pack, dict) and context_pack.get("risk_level"):
        return str(context_pack["risk_level"])
    return "low"


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


def _with_goal_validation(
    tool_name: str,
    tool_input: dict[str, Any],
    result_payload: Any,
    *,
    state: AgentState,
    registry: Any | None = None,
) -> Any:
    if not isinstance(result_payload, dict):
        return result_payload
    goal_state = GoalState.from_metadata_dict(state.context_metadata.get(GOAL_STATE_METADATA_KEY))
    if goal_state is None or not goal_state.expected_operation:
        return result_payload
    semantic_profile = _semantic_profile_for_tool(registry, tool_name)
    expected_operation = str(goal_state.expected_operation or "").strip()
    filesystem_goal = get_filesystem_operation_spec(expected_operation) is not None
    if goal_state.intent != "filesystem_operation" and not filesystem_goal and semantic_profile is None:
        return result_payload

    # This is the runtime-level afterToolCall checkpoint. It does not replace
    # tool success/failure; it adds a second judgment: did the tool result satisfy
    # the user's actual target for this turn?
    validation = validate_goal_completion(
        goal_state=goal_state,
        tool_name=tool_name,
        tool_input=tool_input,
        result_payload=result_payload,
        semantic_profile=semantic_profile,
    )
    validation_payload = validation.to_metadata_dict()
    enriched = {
        **result_payload,
        "goal_validation": validation_payload,
        "goal_completed": validation.completed,
        "goal_advanced": validation.advanced,
    }
    envelope = enriched.get("result_envelope")
    if isinstance(envelope, dict):
        enriched["result_envelope"] = {
            **envelope,
            "goal_validation": validation_payload,
            "goal_completed": validation.completed,
            "goal_advanced": validation.advanced,
        }
    return enriched


def _semantic_profile_for_tool(registry: Any | None, tool_name: str) -> Any | None:
    if registry is None or not tool_name:
        return None
    get_definition = getattr(registry, "get", None)
    if not callable(get_definition):
        return None
    definition = get_definition(tool_name)
    return getattr(definition, "semantic_profile", None) if definition is not None else None


def _generate_final_response(state: AgentState, *, dependencies: AgentGraphDependencies) -> tuple[str, str]:
    unreliable_search_response = _unreliable_external_web_search_response(state)
    if unreliable_search_response is not None:
        return unreliable_search_response, "tool_result_summary_unreliable"
    grounded_filesystem_response = _grounded_filesystem_mutation_response(state)
    if grounded_filesystem_response is not None:
        return grounded_filesystem_response
    if _has_prepared_final_response(state):
        return state.final_response, state.response_mode
    synthesis_messages = external_web_search_synthesis_messages(state)
    if synthesis_messages is not None and dependencies.llm_client is not None:
        try:
            completion = dependencies.llm_client.complete(messages=synthesis_messages)
            return completion.content, "llm_tool_result_summary"
        except Exception:
            pass
    generic_tool_answer = _tool_observation_final_answer_response(state, dependencies=dependencies)
    if generic_tool_answer is not None:
        return generic_tool_answer
    filesystem_synthesis_messages = filesystem_answer_synthesis_messages(state)
    if filesystem_synthesis_messages is not None and dependencies.llm_client is not None:
        try:
            completion = dependencies.llm_client.complete(messages=filesystem_synthesis_messages)
            return completion.content, "llm_filesystem_answer_synthesis"
        except Exception:
            payload = _latest_tool_result_payload(state, FILESYSTEM_SKILL_CAPABILITY)
            fallback = _filesystem_answer_intent_fallback_response(state, payload)
            if fallback is not None:
                return fallback, "filesystem_answer_synthesis_fallback"
    tool_response = tool_result_summary_response(state, dependencies=dependencies)
    if tool_response is not None:
        return tool_response
    if dependencies.llm_client is None:
        return "Agent runtime completed deterministic workflow skeleton.", "deterministic_stub"
    completion = dependencies.llm_client.complete(messages=state.llm_messages)
    return completion.content, "llm"


def _grounded_filesystem_mutation_response(state: AgentState) -> tuple[str, str] | None:
    """Prefer verified file mutation evidence over stale generic assistant text."""

    if state.requested_tool_name != FILESYSTEM_SKILL_CAPABILITY:
        return None
    payload = _latest_tool_result_payload(state, FILESYSTEM_SKILL_CAPABILITY)
    if payload is None:
        return None
    result_payload = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    operation = str(result_payload.get("operation") or "").strip()
    if operation not in {"copy_file", "rename_file"}:
        return None
    if payload.get("status") != "succeeded" or not _tool_result_ok(result_payload):
        return None
    return _filesystem_skill_summary_response(payload), "tool_result_summary"


def _tool_observation_final_answer_response(
    state: AgentState,
    *,
    dependencies: AgentGraphDependencies,
) -> tuple[str, str] | None:
    return complete_tool_observation_final_answer(state, llm_client=dependencies.llm_client)


def tool_result_summary_response(
    state: AgentState,
    *,
    dependencies: AgentGraphDependencies | None = None,
) -> tuple[str, str] | None:
    if state.requested_tool_name == DATABASE_COMPANY_LIST_TOOL:
        payload = _latest_tool_result_payload(state, DATABASE_COMPANY_LIST_TOOL)
        if payload is None:
            return None
        return _database_company_list_summary_response(payload), "tool_result_summary"
    if state.requested_tool_name == LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL:
        payload = _latest_tool_result_payload(state, LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL)
        if payload is None:
            return None
        fallback_response = _company_database_specific_query_fallback_response(state, payload, dependencies=dependencies)
        if fallback_response is not None:
            return fallback_response, "tool_result_summary_fallback"
        insufficient_response = _company_database_specific_query_insufficient_response(state, payload)
        if insufficient_response is not None:
            return insufficient_response, "tool_result_summary_insufficient"
        return _company_database_overview_summary_response(payload), "tool_result_summary"
    if state.requested_tool_name == LOCAL_JOB_SOURCE_OVERVIEW_TOOL:
        payload = _latest_tool_result_payload(state, LOCAL_JOB_SOURCE_OVERVIEW_TOOL)
        if payload is None:
            return None
        context_pack = state.context_metadata.get("context_pack") if isinstance(state.context_metadata, dict) else None
        context_pack = context_pack if isinstance(context_pack, dict) else {}
        sync_policy = context_pack.get("sync_policy") if isinstance(context_pack.get("sync_policy"), dict) else {}
        if sync_policy.get("mode") == "company_board_count":
            return _company_board_count_summary_response(payload), "tool_result_summary"
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
    if state.requested_tool_name == FILESYSTEM_SKILL_CAPABILITY:
        payload = _latest_tool_result_payload(state, FILESYSTEM_SKILL_CAPABILITY)
        if payload is None:
            return None
        fallback = _filesystem_answer_intent_fallback_response(state, payload)
        if fallback is not None:
            return fallback, "filesystem_answer_intent_fallback"
        return _filesystem_skill_summary_response(payload), "tool_result_summary"
    if state.requested_tool_name == EXTERNAL_WEB_SEARCH_TOOL:
        unreliable_search_response = _unreliable_external_web_search_response(state)
        if unreliable_search_response is not None:
            return unreliable_search_response, "tool_result_summary_unreliable"
        payload = _latest_tool_result_payload(state, EXTERNAL_WEB_SEARCH_TOOL)
        if payload is None:
            return None
        return _external_web_search_summary_response(payload), "tool_result_summary"
    return None


def _filesystem_skill_summary_response(payload: dict[str, Any]) -> str:
    result_payload = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    operation = str(result_payload.get("operation") or "").strip()
    spec = get_filesystem_operation_spec(operation)
    arguments = result_payload.get("arguments") if isinstance(result_payload.get("arguments"), dict) else {}
    raw_tool_result = result_payload.get("result") if isinstance(result_payload.get("result"), dict) else {}
    inner_result = raw_tool_result.get("result") if isinstance(raw_tool_result.get("result"), dict) else {}
    ok = payload.get("status") == "succeeded" and _tool_result_ok(result_payload)
    if spec is None:
        envelope = result_payload.get("result_envelope") if isinstance(result_payload.get("result_envelope"), dict) else {}
        # Unknown operation is already an execution failure. Prefer the executor
        # or envelope error summary; never turn an unclassified failure into a
        # misleading user-facing "completed" message.
        summary = str(
            result_payload.get("summary")
            or envelope.get("summary")
            or raw_tool_result.get("error")
            or payload.get("error")
            or ("filesystem Skill 已完成。" if ok else "filesystem Skill 执行失败。")
        )
        return summary

    if not ok:
        error = str(result_payload.get("error") or raw_tool_result.get("error") or payload.get("error") or f"{spec.summary}失败。")
        return f"{spec.summary}失败：{error}"
    if result_payload.get("no_op"):
        return str(result_payload.get("summary") or "没有执行替换：新旧内容相同，文件已经满足当前目标。")
    if spec.goal_kind == "read_content":
        return _filesystem_read_summary(spec=spec, arguments=arguments, inner_result=inner_result)
    if spec.goal_kind == "path_exists":
        path = str(arguments.get("path") or "").strip()
        exists = bool(inner_result.get("exists"))
        return f"文件存在：{path}" if exists else f"文件不存在：{path}"
    if spec.goal_kind == "file_to_file":
        filesystem_trace = result_payload.get("filesystem_trace") if isinstance(result_payload.get("filesystem_trace"), dict) else None
        return _filesystem_file_to_file_summary(spec=spec, arguments=arguments, filesystem_trace=filesystem_trace)

    summary = str(result_payload.get("summary") or raw_tool_result.get("error") or "filesystem Skill 已完成。")
    return summary


def _filesystem_read_summary(*, spec: FilesystemOperationSpec, arguments: dict[str, Any], inner_result: dict[str, Any]) -> str:
    path = str(arguments.get("path") or "").strip()
    content = str(inner_result.get("content") or inner_result.get("stdout") or "")
    if not content.strip():
        return f"已读取文件 {path}，但文件内容为空。" if path else "已读取文件，但文件内容为空。"
    title = f"已读取文件 {path} 的内容：" if path else "已读取文件内容："
    return f"{title}\n\n```text\n{content.rstrip()}\n```"


def filesystem_answer_synthesis_messages(state: AgentState) -> list[dict[str, Any]] | None:
    if state.requested_tool_name != FILESYSTEM_SKILL_CAPABILITY:
        return None
    payload = _latest_tool_result_payload(state, FILESYSTEM_SKILL_CAPABILITY)
    if payload is None:
        return None

    goal_state = GoalState.from_metadata_dict(state.context_metadata.get(GOAL_STATE_METADATA_KEY))
    answer_intent, answer_policy = _filesystem_answer_profile(state, goal_state)
    path = _filesystem_read_path_from_payload(payload) or _filesystem_path_from_goal(goal_state)
    content = _filesystem_read_content_from_payload(payload)
    tool_evidence = _filesystem_final_answer_evidence(payload, content=content)
    if not tool_evidence.strip():
        return None

    policy = {"answer_intent": answer_intent, **answer_policy}
    instruction = (
        "你是 OfferMaster 的主 Agent。filesystem 工具或 Skill 已经执行完成。"
        "工具结果只是 observation，不是最终回答。"
        "不要把 read_file 的原始内容直接当成最终回答；"
        "不要原样展示全文，除非用户明确要求展示原文；"
        "必须回到用户原始目标，判断用户要展示、总结、提取、分析、对比还是确认执行结果。"
        "如果用户明确要求显示全文、读出来、展示原文或 cat 文件，可以展示文件内容；"
        "如果用户要求总结、分析、提取、对比、评价或给建议，必须基于文件内容加工后回答。"
        "不要复述内部工具协议、Tool call、Tool result 或 JSON。用中文回答，表达简洁。"
    )
    evidence_message = (
        f"用户原始问题：{state.user_message}\n"
        f"目标文件：{path}\n"
        f"最终回答策略 JSON：{json.dumps(policy, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "filesystem observation 如下，请把它当作证据，而不是直接当作最终答案：\n"
        f"{tool_evidence}"
    )
    return [
        {"role": "system", "content": instruction, "metadata": {"source": "filesystem_answer_synthesis"}},
        {"role": "user", "content": evidence_message, "metadata": {"source": "filesystem_answer_synthesis"}},
    ]


def _filesystem_answer_profile(state: AgentState, goal_state: GoalState | None) -> tuple[str, dict[str, Any]]:
    if goal_state is not None and goal_state.answer_intent:
        return goal_state.answer_intent, dict(goal_state.answer_policy)
    inferred = _filesystem_answer_intent_from_user_message(state.user_message)
    policy: dict[str, Any] = {"language": "zh-CN"}
    if inferred != "show_content":
        policy["do_not_echo_full_content"] = True
    return inferred, policy


def _filesystem_answer_intent_from_user_message(user_message: str) -> str:
    text = str(user_message or "")
    lowered = text.lower()
    if any(marker in text for marker in ("总结", "概括", "摘要", "归纳", "提炼", "主要内容", "讲了什么", "说了什么", "核心内容")):
        return "summarize_document"
    if any(marker in text for marker in ("提取", "列出", "找出", "抽取", "有哪些", "整理出")):
        return "extract_document_information"
    if any(marker in text for marker in ("分析", "评价", "适合", "匹配", "问题", "建议", "优化点")):
        return "analyze_document"
    if any(marker in lowered for marker in ("cat", "show", "display")) or any(marker in text for marker in ("显示", "展示", "全文", "原文", "读出来", "内容是什么", "告诉我这个文件的内容")):
        return "show_content"
    return "answer_from_filesystem_observation"


def _filesystem_path_from_goal(goal_state: GoalState | None) -> str:
    if goal_state is None:
        return ""
    for key in ("path", "source_path", "target_path"):
        value = goal_state.target.get(key)
        if value:
            return str(value)
    return ""


def _filesystem_final_answer_evidence(payload: dict[str, Any], *, content: str) -> str:
    if content.strip():
        return _truncate_for_filesystem_synthesis(content)
    # Non-read actions still need synthesis to answer from postcheck/status, but
    # the prompt should stay compact and avoid leaking long internal envelopes.
    compact_payload = _compact_filesystem_final_answer_payload(payload)
    return json.dumps(compact_payload, ensure_ascii=False, separators=(",", ":"))


def _compact_filesystem_final_answer_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result_payload = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    raw_tool_result = result_payload.get("result") if isinstance(result_payload.get("result"), dict) else {}
    inner_result = raw_tool_result.get("result") if isinstance(raw_tool_result.get("result"), dict) else {}
    allowed_inner = {
        key: value
        for key, value in inner_result.items()
        if key in {"exists", "path", "source_path", "target_path", "operation", "replacement_count", "no_op"}
    }
    return {
        "status": payload.get("status"),
        "error": payload.get("error") or result_payload.get("error") or raw_tool_result.get("error"),
        "operation": result_payload.get("operation"),
        "summary": result_payload.get("summary") or raw_tool_result.get("summary"),
        "arguments": result_payload.get("arguments") if isinstance(result_payload.get("arguments"), dict) else {},
        "filesystem_trace": _runtime_result_summary_metadata(result_payload).get("filesystem_trace"),
        "result": allowed_inner,
        "goal_validation": result_payload.get("goal_validation") if isinstance(result_payload.get("goal_validation"), dict) else None,
    }


def _filesystem_answer_intent_fallback_response(state: AgentState, payload: dict[str, Any] | None) -> str | None:
    goal_state = GoalState.from_metadata_dict(state.context_metadata.get(GOAL_STATE_METADATA_KEY))
    answer_intent, _answer_policy = _filesystem_answer_profile(state, goal_state)
    if answer_intent in {"show_content", "answer_from_filesystem_observation"}:
        return None
    if payload is not None and not _filesystem_read_result_ok(payload):
        return None
    path = _filesystem_read_path_from_payload(payload or {}) or _filesystem_path_from_goal(goal_state)
    content = _filesystem_read_content_from_payload(payload or {})
    if not content.strip():
        return f"已读取文件 {path}，但文件内容为空，暂无可加工内容。" if path else "已读取文件，但文件内容为空，暂无可加工内容。"
    if path:
        return f"已读取文件 {path}，但当前无法稳定生成最终回答。为避免把原文误当答案返回，本轮不直接展示全文。"
    return "已读取文件，但当前无法稳定生成最终回答。为避免把原文误当答案返回，本轮不直接展示全文。"


def _filesystem_read_result_ok(payload: dict[str, Any]) -> bool:
    result_payload = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    operation = str(result_payload.get("operation") or "").strip()
    return payload.get("status") == "succeeded" and operation == "read_file" and _tool_result_ok(result_payload)


def _filesystem_read_content_from_payload(payload: dict[str, Any]) -> str:
    inner_result = _filesystem_inner_result(payload)
    return str(inner_result.get("content") or inner_result.get("stdout") or "")


def _filesystem_read_path_from_payload(payload: dict[str, Any]) -> str:
    result_payload = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    arguments = result_payload.get("arguments") if isinstance(result_payload.get("arguments"), dict) else {}
    return str(arguments.get("path") or "").strip()


def _filesystem_inner_result(payload: dict[str, Any]) -> dict[str, Any]:
    result_payload = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    raw_tool_result = result_payload.get("result") if isinstance(result_payload.get("result"), dict) else {}
    return raw_tool_result.get("result") if isinstance(raw_tool_result.get("result"), dict) else {}


def _truncate_for_filesystem_synthesis(content: str, *, limit: int = 12000) -> str:
    text = str(content or "")
    if len(text) <= limit:
        return text
    # Keep the synthesis prompt bounded; the original tool result remains in
    # structured runtime metadata, while the LLM only needs enough evidence to
    # answer the user's summary request.
    return f"{text[:limit]}\n\n[文件内容过长，已截断用于本轮摘要生成。]"


def _filesystem_file_to_file_summary(
    *,
    spec: FilesystemOperationSpec,
    arguments: dict[str, Any],
    filesystem_trace: dict[str, Any] | None = None,
) -> str:
    source_key = spec.required_args[0] if spec.required_args else "src"
    source = str(arguments.get(source_key) or arguments.get("path") or "").strip()
    target = str(arguments.get(spec.result_path_arg) or "").strip()
    if source and target:
        postcheck = filesystem_trace.get("postcheck") if isinstance(filesystem_trace, dict) and isinstance(filesystem_trace.get("postcheck"), dict) else {}
        if postcheck.get("completed") is True:
            return f"已完成并复核：{spec.summary} {source} -> {target}"
        return f"已完成：{spec.summary} {source} -> {target}"
    return f"已完成：{spec.summary}"


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


def _database_company_list_summary_response(payload: dict[str, Any]) -> str:
    tool_result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    result = tool_result.get("result") if isinstance(tool_result.get("result"), dict) else {}
    ok = payload.get("status") == "succeeded" and _tool_result_ok(tool_result)
    if not ok:
        error = payload.get("error") or tool_result.get("error") or result.get("message")
        return f"本地公司列表读取失败：{error or 'unknown error'}。"

    total_count = _safe_count(result.get("total_count"))
    display_count = _safe_count(result.get("count"))
    companies = result.get("companies") if isinstance(result.get("companies"), list) else []
    response = f"当前本地数据库去重公司共 {total_count} 家，本次展示 {display_count} 家："
    table = _database_company_list_markdown_table(companies)
    if table:
        response += f"\n\n{table}"
    else:
        response += "\n\n没有可展示的公司记录。"
    response += "\n\n说明：这里汇总正式企业、正式岗位、岗位线索和招聘来源信号中的公司名称；只读展示，不会修改数据库。"
    return response


def _database_company_list_markdown_table(companies: list[Any]) -> str:
    if not companies:
        return ""
    lines = [
        "| 公司 | 企业档案 | 正式岗位 | 岗位线索 | 招聘来源 | 已有记录 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for company in companies:
        if not isinstance(company, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_table_cell(company.get("company_name")),
                    "有" if company.get("has_profile") else "无",
                    str(_safe_count(company.get("job_count"))),
                    str(_safe_count(company.get("lead_count"))),
                    str(_safe_count(company.get("signal_count"))),
                    str(_safe_count(company.get("total_record_count"))),
                ]
            )
            + " |"
        )
    return "\n".join(lines) if len(lines) > 2 else ""


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


def _company_database_specific_query_fallback_response(
    state: AgentState,
    payload: dict[str, Any],
    *,
    dependencies: AgentGraphDependencies | None,
) -> str | None:
    targets = _specific_company_query_names_from_state(state)
    if not targets or dependencies is None or dependencies.db_session is None:
        return None
    details = _local_company_specific_details(dependencies.db_session, targets)
    if not details.get("has_any_detail"):
        return None
    return _company_specific_details_response(targets, details)


def _local_company_specific_details(session: Session, targets: list[str]) -> dict[str, Any]:
    from sqlalchemy import select

    from app.domains.jobs.models import Company, Job, JobLead, RecruitingSignal

    exact_details = _local_company_specific_details_with_match_mode(
        session,
        targets,
        Company=Company,
        Job=Job,
        JobLead=JobLead,
        RecruitingSignal=RecruitingSignal,
        select=select,
        exact=True,
    )
    if exact_details.get("has_any_detail"):
        return exact_details
    return _local_company_specific_details_with_match_mode(
        session,
        targets,
        Company=Company,
        Job=Job,
        JobLead=JobLead,
        RecruitingSignal=RecruitingSignal,
        select=select,
        exact=False,
    )


def _local_company_specific_details_with_match_mode(
    session: Session,
    targets: list[str],
    *,
    Company: Any,
    Job: Any,
    JobLead: Any,
    RecruitingSignal: Any,
    select: Any,
    exact: bool,
) -> dict[str, Any]:
    company_filter = _company_target_filter(Company.name, targets, exact=exact)
    lead_filter = _company_target_filter(JobLead.company_name, targets, exact=exact)
    signal_filter = _company_target_filter(RecruitingSignal.company_name, targets, exact=exact)

    companies = list(session.scalars(select(Company).where(company_filter).order_by(Company.name.asc()).limit(5)))
    jobs = list(
        session.scalars(
            select(Job)
            .join(Company, Job.company_id == Company.id)
            .where(company_filter)
            .order_by(Job.updated_at.desc())
            .limit(5)
        )
    )
    leads = list(session.scalars(select(JobLead).where(lead_filter).order_by(JobLead.updated_at.desc()).limit(5)))
    signals = list(
        session.scalars(
            select(RecruitingSignal).where(signal_filter).order_by(RecruitingSignal.updated_at.desc()).limit(5)
        )
    )
    return {
        "companies": companies,
        "jobs": jobs,
        "leads": leads,
        "signals": signals,
        "has_any_detail": bool(companies or jobs or leads or signals),
    }


def _company_target_filter(column: Any, targets: list[str], *, exact: bool) -> Any:
    from sqlalchemy import or_

    conditions = []
    for target in targets:
        cleaned = str(target or "").strip()
        if cleaned:
            conditions.append(column.ilike(cleaned if exact else f"%{cleaned}%"))
    return or_(*conditions) if conditions else column == "__no_company_target__"


def _company_specific_details_response(targets: list[str], details: dict[str, Any]) -> str:
    target_text = "、".join(targets)
    companies = details.get("companies") if isinstance(details.get("companies"), list) else []
    jobs = details.get("jobs") if isinstance(details.get("jobs"), list) else []
    leads = details.get("leads") if isinstance(details.get("leads"), list) else []
    signals = details.get("signals") if isinstance(details.get("signals"), list) else []

    lines = [f"我查了本地已有来源中关于 {target_text} 的信息："]
    lines.append(_formal_company_profile_line(companies))
    lines.append(_formal_jobs_line(jobs))
    lines.append(_job_leads_line(leads))
    lines.append(_recruiting_signals_line(signals))
    lines.append("说明：这些是本地已有记录，不等于完整企业档案；如果要补齐公司业务、官网岗位和最新进展，还需要继续联网核对。")
    return "\n".join(line for line in lines if line)


def _formal_company_profile_line(companies: list[Any]) -> str:
    if not companies:
        return "- 正式企业档案：未找到。"
    samples = []
    for company in companies[:3]:
        parts = [_object_text(company, "name")]
        industry = _object_text(company, "industry")
        city = _object_text(company, "city")
        website = _object_text(company, "website_url")
        if industry:
            parts.append(f"行业：{industry}")
        if city:
            parts.append(f"城市：{city}")
        if website:
            parts.append(f"官网：{website}")
        samples.append("；".join(part for part in parts if part))
    return f"- 正式企业档案：找到 {len(companies)} 条，" + "；".join(samples) + "。"


def _formal_jobs_line(jobs: list[Any]) -> str:
    if not jobs:
        return "- 正式岗位：未找到。"
    samples = []
    for job in jobs[:3]:
        title = _object_text(job, "title") or "未命名岗位"
        city = _object_text(job, "city")
        job_type = _object_text(job, "job_type")
        source_url = _object_text(job, "source_url")
        parts = [title]
        if job_type:
            parts.append(job_type)
        if city:
            parts.append(f"城市：{city}")
        if source_url:
            parts.append(f"链接：{source_url}")
        samples.append("（" + "；".join(parts[1:]) + "）" if len(parts) > 1 else title)
        if len(parts) > 1:
            samples[-1] = f"{title}{samples[-1]}"
    return f"- 正式岗位 {len(jobs)} 条：" + "；".join(samples) + "。"


def _job_leads_line(leads: list[Any]) -> str:
    if not leads:
        return "- 岗位线索：未找到。"
    samples = []
    for lead in leads[:3]:
        title = _object_text(lead, "title") or "未命名线索"
        parts = []
        for label, attr in (("类型", "job_type"), ("届别", "graduation_year"), ("城市", "city"), ("方向", "job_direction")):
            value = _object_text(lead, attr)
            if value:
                parts.append(f"{label}：{value}")
        status = _object_value_text(getattr(lead, "verification_status", None))
        if status:
            parts.append(f"状态：{status}")
        source_name = _object_text(getattr(lead, "source", None), "name")
        if source_name:
            parts.append(f"来源：{_job_source_display_name(source_name)}")
        source_url = _object_text(lead, "source_url") or _object_text(lead, "apply_url") or _object_text(lead, "verified_url")
        if source_url:
            parts.append(f"链接：{source_url}")
        samples.append(f"{title}（" + "；".join(parts) + "）" if parts else title)
    return f"- 岗位线索 {len(leads)} 条：" + "；".join(samples) + "。"


def _recruiting_signals_line(signals: list[Any]) -> str:
    if not signals:
        return "- 校招来源：未找到。"
    samples = []
    for signal in signals[:3]:
        signal_type = _object_value_text(getattr(signal, "signal_type", None)) or "招聘信号"
        parts = []
        graduation_year = _object_text(signal, "graduation_year")
        if graduation_year:
            parts.append(f"届别：{graduation_year}")
        original_source = _object_text(signal, "original_source")
        if original_source:
            parts.append(f"原始来源：{original_source}")
        trust_level = _object_value_text(getattr(signal, "trust_level", None))
        if trust_level:
            parts.append(f"可信度：{trust_level}")
        status = _object_value_text(getattr(signal, "status", None))
        if status:
            parts.append(f"状态：{status}")
        source_url = _object_text(signal, "source_url")
        if source_url:
            parts.append(f"链接：{source_url}")
        samples.append(f"{signal_type}（" + "；".join(parts) + "）" if parts else signal_type)
    return f"- 校招来源 {len(signals)} 条：" + "；".join(samples) + "。"


def _object_text(obj: Any, attr: str) -> str:
    if obj is None:
        return ""
    return str(getattr(obj, attr, "") or "").strip()


def _object_value_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip()


def _company_database_specific_query_insufficient_response(state: AgentState, payload: dict[str, Any]) -> str | None:
    targets = _specific_company_query_names_from_state(state)
    if not targets:
        return None
    matched_names = _matching_company_overview_names(payload, targets)
    target_text = "、".join(targets)
    if matched_names:
        matched_text = "、".join(matched_names)
        return (
            f"我理解你要查的是 {target_text} 这家公司，不是全库概览。"
            f"当前这次工具只返回了企业库概览，只能确认概览数据里出现了 {matched_text}，"
            "但没有提供它的专属企业档案、岗位详情或来源明细，所以我不能把全库概览当成答案。"
            "需要继续查岗位线索、校招来源或公开信息后，再给你汇总。"
        )
    return (
        f"我理解你要查的是 {target_text} 这家公司，不是全库概览。"
        "当前这次工具只返回了企业库概览，没有提供目标公司的专属信息，"
        "所以我不能把全库概览当成答案。需要继续查其他本地来源或公开信息后，再给你汇总。"
    )


def _specific_company_query_names_from_state(state: AgentState) -> list[str]:
    names = _expected_company_names_from_state(state)
    names.extend(_specific_company_query_names_from_user_message(state.user_message))
    return _dedupe_company_names(names)


def _specific_company_query_names_from_user_message(message: str) -> list[str]:
    text = str(message or "").strip()
    if not text:
        return []
    patterns = (
        r"关于(?P<name>[\w\u4e00-\u9fff·.&（）()\- ]{1,40}?)(?:这个|这家)?(?:公司|企业)",
        r"(?:查一下|看一下|看看|了解一下|检索)(?P<name>[\w\u4e00-\u9fff·.&（）()\- ]{1,40}?)(?:这个|这家)?(?:公司|企业)",
        r"(?P<name>[\w\u4e00-\u9fff·.&（）()\- ]{1,40}?)(?:这个|这家)?(?:公司|企业)(?:的信息|资料|有什么|有哪些)",
    )
    names: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            name = _clean_specific_company_query_name(match.group("name"))
            if name:
                names.append(name)
    return _dedupe_company_names(names)


def _clean_specific_company_query_name(name: Any) -> str:
    cleaned = str(name or "").strip(" ：:，,。！？?！ 的")
    cleaned = re.sub(r"^(?:你给我|帮我|请|一下|看一下|查一下|看看|了解一下)", "", cleaned).strip()
    cleaned = re.sub(r"^(?:数据库中|数据库里|本地数据库中|本地数据库里|本地库中|本地库里|公司库中|企业库中|关于)", "", cleaned).strip()
    cleaned = cleaned.strip(" ：:，,。！？?！ 的")
    if not cleaned:
        return ""
    generic_terms = {"数据库", "本地数据库", "公司库", "企业库", "公司", "企业", "哪些", "多少", "信息"}
    return "" if cleaned in generic_terms else cleaned


def _dedupe_company_names(names: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        cleaned = _clean_specific_company_query_name(name)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            deduped.append(cleaned)
            seen.add(key)
    return deduped


def _matching_company_overview_names(payload: dict[str, Any], targets: list[str]) -> list[str]:
    target_keys = [target.casefold() for target in targets if target]
    if not target_keys:
        return []
    result = _company_database_payload_result(payload)
    candidate_names: list[str] = []
    for key in ("sample_companies", "sample_lead_companies", "sample_signal_companies"):
        values = result.get(key)
        if isinstance(values, list):
            candidate_names.extend(str(value).strip() for value in values if str(value).strip())
    rows = result.get("company_rows")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                name = str(row.get("company_name") or "").strip()
                if name:
                    candidate_names.append(name)
    matches: list[str] = []
    seen: set[str] = set()
    for candidate in candidate_names:
        candidate_key = candidate.casefold()
        if any(target_key in candidate_key or candidate_key in target_key for target_key in target_keys):
            if candidate_key not in seen:
                matches.append(candidate)
                seen.add(candidate_key)
    return matches


def _company_database_payload_result(payload: dict[str, Any]) -> dict[str, Any]:
    tool_result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    result = tool_result.get("result") if isinstance(tool_result.get("result"), dict) else {}
    return result if isinstance(result, dict) else {}


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


def _company_board_count_summary_response(payload: dict[str, Any]) -> str:
    tool_result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    result = tool_result.get("result") if isinstance(tool_result.get("result"), dict) else {}
    ok = payload.get("status") == "succeeded" and _tool_result_ok(tool_result)
    if not ok:
        error = payload.get("error") or tool_result.get("error") or result.get("message")
        return f"公司展览公司数暂时无法读取：{error or '数据源未返回有效结果'}。我没有用本地历史企业表替代这个数字。"
    company_board = result.get("company_board") if isinstance(result.get("company_board"), dict) else {}
    count = company_board.get("company_count")
    if count is None:
        return "公司展览数据源没有返回公司总数，因此我不能给出猜测数字。"
    label = str(company_board.get("label") or "公司展览当前公司库")
    return f"公司展览当前的公司数是 { _safe_count(count) } 家，统计口径：{label}。"


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
