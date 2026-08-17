from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import date, datetime
import re
from typing import Any
from uuid import uuid4

try:  # Keep LangGraph's upstream pending-deprecation noise out of targeted runtime tests.
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
except Exception:  # pragma: no cover - only relevant when langchain_core changes its warning location.
    LangChainPendingDeprecationWarning = Warning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning, module=r"langgraph\..*")

from app.agent_runtime.checkpoints import AgentCheckpointStore
from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolNextAction, AgentToolRuntimeGuard
from app.agent_runtime.memory.compaction import CompactionConfig
from app.agent_runtime.memory.context_builder import ContextBuildConfig, MemoryContextBuilder
from app.agent_runtime.memory.skill_repository import AgentSkillRepository
from app.agent_runtime.state import AgentState
from app.agent_runtime.tool_permissions import AgentToolPermissionPolicy
from app.agent_runtime.tool_registry import AgentToolRegistry
from app.domains.automation.models import ApprovalRequest, ToolCallStatus, WorkflowRun, WorkflowRunStatus, utc_now
from app.domains.automation.schemas import ApprovalRequestCreate, ToolCallLogCreate, WorkflowRunCreate
from app.domains.automation.service import AutomationService
from app.domains.conversations.models import AgentMessageRole
from app.domains.conversations.schemas import AgentMessageCreate
from app.domains.conversations.service import ConversationService
from sqlalchemy.orm import Session


AGENT_GRAPH_NODE_ORDER = ["build_context", "plan_or_reply", "maybe_tool", "wait_confirmation", "final_response"]


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

    def with_registry(self, registry: AgentToolRegistry) -> AgentGraphDependencies:
        return replace(self, registry=registry)


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
    return _run_until_response_ready(state, command=command, workflow=workflow, dependencies=dependencies)


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
) -> AgentPreparedResponse:
    state = _build_context_node(state, dependencies=dependencies)
    _save_step(workflow, state, dependencies)

    command = _auto_select_tool_command(command, state=state, registry=dependencies.registry)
    if command.requested_tool_name:
        state = state.with_updates(requested_tool_name=command.requested_tool_name, source_type=command.source_type)

    state = state.with_updates(current_step="plan_or_reply")
    _save_step(workflow, state, dependencies)

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
    return state.with_updates(
        current_step="build_context",
        latest_summary_id=metadata.get("summary_id"),
        loaded_session_history_ids=list(metadata.get("loaded_session_history_ids") or []),
        loaded_memory_ids=list(metadata.get("loaded_memory_ids") or []),
        loaded_skill_ids=list(metadata.get("loaded_skill_ids") or []),
        need_compaction=bool(metadata.get("need_compaction") or False),
        token_estimate=int(metadata.get("token_estimate") or 0),
        llm_messages=built.llm_messages,
        context_metadata=metadata,
    )


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
    if definition is None or definition.handler is None or dependencies.db_session is None:
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
        return state.with_updates(
            current_step="maybe_tool",
            guard_result=guard_payload,
            tool_call_ids=[*state.tool_call_ids, tool_call.id],
        )

    try:
        raw_result = definition.handler(dependencies.db_session, **tool_input)
        result_payload = _jsonable(raw_result)
        tool_ok = _tool_result_ok(result_payload)
        tool_error = None if tool_ok else _tool_result_error(result_payload)
        tool_call = dependencies.automation_service.record_tool_call(
            ToolCallLogCreate(
                workflow_run_id=state.workflow_run_id,
                tool_name=command.requested_tool_name or "",
                tool_group="agent",
                status=ToolCallStatus.SUCCEEDED if tool_ok else ToolCallStatus.FAILED,
                input_payload=tool_input,
                output_payload={"guard_result": guard_payload, "execution": "handler", "result": result_payload},
                error=tool_error,
            )
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
                    "execution": "handler",
                    "error_type": exc.__class__.__name__,
                },
                error=str(exc),
            )
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
    return state.with_updates(
        current_step="maybe_tool",
        guard_result=guard_payload,
        tool_call_ids=[*state.tool_call_ids, tool_call.id],
        llm_messages=[*state.llm_messages, *tool_messages],
    )


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
    if command.requested_tool_name in {"sessions_search", "memory_search"}:
        return {"query": state.user_message, "limit": 10}
    if command.requested_tool_name == "sessions_history":
        return {"session_key": state.session_id, "window_before": 5, "window_after": 5}
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

    if _looks_like_xiaohongshu_search(command.user_message) and registry.get("xiaohongshu-mcp.search_feeds") is not None:
        return replace(
            command,
            requested_tool_name="xiaohongshu-mcp.search_feeds",
            source_type="xiaohongshu_note",
            tool_input={"keyword": _xiaohongshu_keyword(command.user_message)},
        )

    return command


_WEIXIN_ARTICLE_URL_RE = re.compile(r"https?://mp\.weixin\.qq\.com/[^\s)）>\]]+", re.IGNORECASE)
_XIAOHONGSHU_HOST_RE = re.compile(
    r"xiaohongshu\.com|xhslink\.com|\u5c0f\u7ea2\u4e66|\u5c0f\u7d05\u66f8|\u7ea2\u4e66|\u7d05\u66f8",
    re.IGNORECASE,
)
_XIAOHONGSHU_FEED_ID_RE = re.compile(r"(?:feed_id|note_id|item_id)\s*[=:\uff1a]\s*([a-zA-Z0-9_-]+)")
_XIAOHONGSHU_XSEC_RE = re.compile(r"xsec_token\s*[=:\uff1a]\s*([^\s&\uff0c,]+)")


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


def _looks_like_xiaohongshu_search(text: str) -> bool:
    if _XIAOHONGSHU_HOST_RE.search(text) is None:
        return False
    intent_keywords = [
        "\u641c\u7d22",  # search
        "\u67e5",  # look up
        "\u627e",  # find
        "\u83b7\u53d6",  # fetch
        "\u79cb\u62db",  # autumn recruiting
        "\u6821\u62db",  # campus recruiting
        "\u62db\u8058",  # recruiting
        "\u5c97\u4f4d",  # position
        "Java",
        "Agent",
        "AI",
    ]
    return any(keyword in text for keyword in intent_keywords)


def _xiaohongshu_keyword(text: str) -> str:
    keyword = re.sub(r"https?://\S+", " ", text).strip()
    keyword = re.sub(r"\s+", " ", keyword)
    return keyword or text.strip()


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
    return {
        "role": "assistant",
        "content": message.content_text or "",
        "metadata": {"message_id": message.id, "source": "tool_transcript"},
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


def _generate_final_response(state: AgentState, *, dependencies: AgentGraphDependencies) -> tuple[str, str]:
    if dependencies.llm_client is None:
        return "Agent runtime completed deterministic workflow skeleton.", "deterministic_stub"
    completion = dependencies.llm_client.complete(messages=state.llm_messages)
    return completion.content, "llm"


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
