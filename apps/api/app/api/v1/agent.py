from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.agent_runtime.checkpoints import AgentCheckpointStore
from app.agent_runtime.graph_factory import (
    AgentGraphDependencies,
    AgentRunCommand,
    continue_agent_workflow_after_approval,
    finalize_agent_workflow_response,
    prepare_agent_workflow_response,
    run_agent_workflow,
)
from app.agent_runtime.guardrails import AgentToolRuntimeGuard
from app.agent_runtime.memory.compaction import CompactionConfig
from app.agent_runtime.memory.skill_repository import AgentSkillRepository
from app.agent_runtime.tool_registry import create_default_agent_tool_registry, create_mcp_agent_tool_definitions
from app.core.config import get_settings
from app.db.session import get_db_session
from app.domains.agent_memory.repository import AgentMemoryRepository
from app.domains.automation.repository import (
    ApprovalRequestRepository,
    ToolCallLogRepository,
    WorkflowCheckpointRepository,
    WorkflowRunRepository,
)
from app.domains.automation.models import ApprovalRequest, utc_now
from app.domains.automation.schemas import ApprovalRequestRead
from app.domains.automation.service import AutomationService
from app.domains.conversations.models import AgentMessageRole
from app.domains.conversations.repository import ConversationRepository
from app.domains.conversations.schemas import (
    AgentCompactRequest,
    AgentCompactResponse,
    AgentChatTurnResponse,
    AgentContextSummaryRead,
    AgentMessageCreate,
    AgentMessageListResponse,
    AgentMessageRead,
    AgentSessionCreate,
    AgentSessionListResponse,
    AgentSessionRead,
    AgentSessionUpdate,
    AgentUserMessageRequest,
)
from app.domains.conversations.service import ConversationService
from app.infrastructure.llm.chat_client import LLMChatClient
from app.infrastructure.llm.client import build_llm_runtime_config
from app.mcp_gateway.content_source_client import ContentSourceMCPClient
from app.mcp_gateway.client import HttpMCPGatewayClient


router = APIRouter(prefix="/api/v1/agent", tags=["agent"])

ASSISTANT_STUB_REPLY = "我已经记录这条消息。下一步会在记忆系统接入后构建上下文。"


class AgentApprovalDecisionRequest(BaseModel):
    decision_reason: str | None = None


class AgentApprovalDecisionResponse(BaseModel):
    approval: ApprovalRequestRead
    assistant_message: AgentMessageRead
    context_metadata: dict[str, object]


@router.post("/sessions", response_model=AgentSessionRead, status_code=status.HTTP_201_CREATED)
def create_agent_session(
    request: AgentSessionCreate,
    session: Session = Depends(get_db_session),
) -> AgentSessionRead:
    agent_session = _conversation_service(session).create_session(
        title=request.title,
        primary_intent=request.primary_intent,
        metadata_json=request.metadata_json,
    )
    session.commit()
    return AgentSessionRead.model_validate(agent_session)


@router.get("/sessions", response_model=AgentSessionListResponse)
def list_agent_sessions(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    include_archived: bool = False,
    session: Session = Depends(get_db_session),
) -> AgentSessionListResponse:
    sessions = _conversation_service(session).list_sessions(limit=limit, offset=offset, include_archived=include_archived)
    return AgentSessionListResponse(items=[AgentSessionRead.model_validate(item) for item in sessions])


@router.get("/sessions/{session_id}", response_model=AgentSessionRead)
def get_agent_session(
    session_id: str,
    session: Session = Depends(get_db_session),
) -> AgentSessionRead:
    try:
        agent_session = _conversation_service(session).get_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AgentSessionRead.model_validate(agent_session)


@router.patch("/sessions/{session_id}", response_model=AgentSessionRead)
def update_agent_session(
    session_id: str,
    request: AgentSessionUpdate,
    session: Session = Depends(get_db_session),
) -> AgentSessionRead:
    try:
        agent_session = _conversation_service(session).update_session(
            session_id,
            title=request.title,
            primary_intent=request.primary_intent,
            metadata_json=request.metadata_json,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    return AgentSessionRead.model_validate(agent_session)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_agent_session(
    session_id: str,
    session: Session = Depends(get_db_session),
) -> Response:
    try:
        _conversation_service(session).archive_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/sessions/{session_id}/compact", response_model=AgentCompactResponse)
def compact_agent_session(
    session_id: str,
    request: AgentCompactRequest,
    session: Session = Depends(get_db_session),
) -> AgentCompactResponse:
    try:
        result = _conversation_service(session).compact_session(
            session_id,
            CompactionConfig(
                context_window=request.context_window,
                reserve_tokens=request.reserve_tokens,
                keep_recent_tokens=request.keep_recent_tokens,
            ),
        )
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "not found" in message else 400
        raise HTTPException(status_code=status_code, detail=message) from exc

    session.commit()
    return AgentCompactResponse(
        summary=AgentContextSummaryRead.model_validate(result.summary),
        covered_message_count=result.covered_message_count,
        first_kept_message_id=result.first_kept_message_id,
        token_estimate_before=result.token_estimate_before,
        token_estimate_after=result.token_estimate_after,
        should_compact=result.should_compact,
    )


@router.post(
    "/sessions/{session_id}/messages",
    response_model=AgentChatTurnResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_agent_message(
    session_id: str,
    request: AgentUserMessageRequest,
    session: Session = Depends(get_db_session),
) -> AgentChatTurnResponse:
    service = _conversation_service(session)
    try:
        dependencies = _agent_graph_dependencies(session, service)
        workflow_result = run_agent_workflow(
            _agent_run_command_from_request(session_id, request),
            dependencies=dependencies,
        )
        assistant_content = (
            workflow_result.state.final_response
            if workflow_result.state.response_mode == "llm" and workflow_result.state.final_response
            else ASSISTANT_STUB_REPLY
        )
        user_message = service.append_message(
            session_id,
            AgentMessageCreate(
                role=AgentMessageRole.USER,
                content_text=request.content_text,
                visible_content_text=request.content_text,
                agent_run_id=workflow_result.state.agent_run_id,
                workflow_run_id=workflow_result.workflow_run_id,
                metadata_json=request.metadata_json,
            ),
        )
        assistant_message = service.append_message(
            session_id,
            AgentMessageCreate(
                role=AgentMessageRole.ASSISTANT,
                content_text=assistant_content,
                visible_content_text=assistant_content,
                parent_message_id=user_message.id,
                agent_run_id=workflow_result.state.agent_run_id,
                workflow_run_id=workflow_result.workflow_run_id,
                metadata_json={
                    "response_mode": workflow_result.state.response_mode,
                    "context_metadata": _context_metadata_from_state(workflow_result.state),
                },
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    session.commit()
    return AgentChatTurnResponse(
        user_message=AgentMessageRead.model_validate(user_message),
        assistant_message=AgentMessageRead.model_validate(assistant_message),
    )


@router.post("/sessions/{session_id}/messages/stream")
def create_agent_message_stream(
    session_id: str,
    request: AgentUserMessageRequest,
    session: Session = Depends(get_db_session),
) -> StreamingResponse:
    def event_stream() -> Iterator[str]:
        service = _conversation_service(session)
        try:
            dependencies = _agent_graph_dependencies(session, service)
            prepared = prepare_agent_workflow_response(
                _agent_run_command_from_request(session_id, request),
                dependencies=dependencies,
            )
            user_message = service.append_message(
                session_id,
                AgentMessageCreate(
                    role=AgentMessageRole.USER,
                    content_text=request.content_text,
                    visible_content_text=request.content_text,
                    agent_run_id=prepared.state.agent_run_id,
                    workflow_run_id=prepared.workflow_run_id,
                    metadata_json=request.metadata_json,
                ),
            )
            session.commit()
            yield _sse_event("user_message", {"message": AgentMessageRead.model_validate(user_message).model_dump(mode="json")})

            if prepared.state.current_step == "wait_confirmation":
                approval = session.get(ApprovalRequest, prepared.state.approval_request_id)
                if approval is None:
                    raise ValueError("Agent is waiting for user confirmation, but approval request was not found.")
                yield _sse_event("approval_required", _approval_required_payload(approval, prepared.state))
                return

            chunks: list[str] = []
            response_mode = "llm_stream"
            if dependencies.llm_client is None:
                response_mode = "deterministic_stub"
                chunk_iterable = [ASSISTANT_STUB_REPLY]
            elif hasattr(dependencies.llm_client, "stream_complete"):
                chunk_iterable = dependencies.llm_client.stream_complete(messages=prepared.state.llm_messages)
            else:
                response_mode = "llm"
                chunk_iterable = [dependencies.llm_client.complete(messages=prepared.state.llm_messages).content]

            for chunk in chunk_iterable:
                if not chunk:
                    continue
                chunks.append(str(chunk))
                yield _sse_event("token", {"content": str(chunk)})

            assistant_content = "".join(chunks).strip() or ASSISTANT_STUB_REPLY
            workflow_result = finalize_agent_workflow_response(
                prepared.state,
                final_response=assistant_content,
                response_mode=response_mode,
                dependencies=dependencies,
            )
            assistant_message = service.append_message(
                session_id,
                AgentMessageCreate(
                    role=AgentMessageRole.ASSISTANT,
                    content_text=assistant_content,
                    visible_content_text=assistant_content,
                    parent_message_id=user_message.id,
                    agent_run_id=workflow_result.state.agent_run_id,
                    workflow_run_id=workflow_result.workflow_run_id,
                    metadata_json={
                        "response_mode": workflow_result.state.response_mode,
                        "context_metadata": _context_metadata_from_state(workflow_result.state),
                    },
                ),
            )
            session.commit()
            yield _sse_event(
                "done",
                {
                    "assistant_message": AgentMessageRead.model_validate(assistant_message).model_dump(mode="json"),
                    "context_metadata": _context_metadata_from_state(workflow_result.state),
                },
            )
        except ValueError as exc:
            session.rollback()
            yield _sse_event("error", {"message": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive streaming boundary.
            session.rollback()
            yield _sse_event("error", {"message": f"Agent stream failed: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/approvals/{approval_request_id}/approve", response_model=AgentApprovalDecisionResponse)
def approve_agent_approval(
    approval_request_id: str,
    request: AgentApprovalDecisionRequest,
    session: Session = Depends(get_db_session),
) -> AgentApprovalDecisionResponse:
    service = _conversation_service(session)
    try:
        dependencies = _agent_graph_dependencies(session, service)
        workflow_result = continue_agent_workflow_after_approval(
            approval_request_id,
            approved=True,
            decision_reason=request.decision_reason,
            dependencies=dependencies,
        )
        assistant_message = _append_assistant_message_from_state(service, workflow_result.state)
        session.flush()
        approval = session.get(ApprovalRequest, approval_request_id)
        if approval is None:
            raise ValueError(f"Approval request not found: {approval_request_id}")
        _apply_approval_decision_for_response(approval, status="approved", decision=request.decision_reason)
        session.flush()
        session.refresh(approval)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session.commit()
    return AgentApprovalDecisionResponse(
        approval=ApprovalRequestRead.model_validate(approval),
        assistant_message=AgentMessageRead.model_validate(assistant_message),
        context_metadata=_context_metadata_from_state(workflow_result.state),
    )


@router.post("/approvals/{approval_request_id}/reject", response_model=AgentApprovalDecisionResponse)
def reject_agent_approval(
    approval_request_id: str,
    request: AgentApprovalDecisionRequest,
    session: Session = Depends(get_db_session),
) -> AgentApprovalDecisionResponse:
    service = _conversation_service(session)
    try:
        dependencies = _agent_graph_dependencies(session, service)
        workflow_result = continue_agent_workflow_after_approval(
            approval_request_id,
            approved=False,
            decision_reason=request.decision_reason,
            dependencies=dependencies,
        )
        assistant_message = _append_assistant_message_from_state(service, workflow_result.state)
        session.flush()
        approval = session.get(ApprovalRequest, approval_request_id)
        if approval is None:
            raise ValueError(f"Approval request not found: {approval_request_id}")
        _apply_approval_decision_for_response(approval, status="rejected", decision=request.decision_reason)
        session.flush()
        session.refresh(approval)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session.commit()
    return AgentApprovalDecisionResponse(
        approval=ApprovalRequestRead.model_validate(approval),
        assistant_message=AgentMessageRead.model_validate(assistant_message),
        context_metadata=_context_metadata_from_state(workflow_result.state),
    )


@router.get("/sessions/{session_id}/messages", response_model=AgentMessageListResponse)
def list_agent_messages(
    session_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    before_message_id: str | None = None,
    session: Session = Depends(get_db_session),
) -> AgentMessageListResponse:
    try:
        messages = _conversation_service(session).list_messages(
            session_id,
            limit=limit,
            before_message_id=before_message_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AgentMessageListResponse(items=[AgentMessageRead.model_validate(message) for message in messages])


def _agent_run_command_from_request(session_id: str, request: AgentUserMessageRequest) -> AgentRunCommand:
    return AgentRunCommand(
        session_id=session_id,
        user_message=request.content_text,
        requested_tool_name=request.requested_tool_name,
        source_type=request.source_type,
        user_confirmed=request.user_confirmed,
        tool_input=request.tool_input or {},
    )


def _append_assistant_message_from_state(service: ConversationService, state) -> object:
    assistant_content = state.final_response or ASSISTANT_STUB_REPLY
    return service.append_message(
        state.session_id,
        AgentMessageCreate(
            role=AgentMessageRole.ASSISTANT,
            content_text=assistant_content,
            visible_content_text=assistant_content,
            agent_run_id=state.agent_run_id,
            workflow_run_id=state.workflow_run_id,
            metadata_json={
                "response_mode": state.response_mode,
                "context_metadata": _context_metadata_from_state(state),
            },
        ),
    )


def _approval_required_payload(approval: ApprovalRequest, state) -> dict[str, object]:
    guard_result = state.guard_result if isinstance(state.guard_result, dict) else {}
    error_details = guard_result.get("error_details") if isinstance(guard_result.get("error_details"), dict) else {}
    payload = approval.payload if isinstance(approval.payload, dict) else {}
    tool_name = str(payload.get("requested_tool_name") or approval.action_type)
    return {
        "approval": ApprovalRequestRead.model_validate(approval).model_dump(mode="json"),
        "approval_request_id": approval.id,
        "workflow_run_id": approval.workflow_run_id,
        "tool_name": tool_name,
        "reason": guard_result.get("reason") or approval.prompt,
        "user_message": guard_result.get("user_message") or approval.prompt,
        "permission_decision": error_details.get("permission_decision"),
        "skill_ids": error_details.get("skill_ids") or [],
        "context_metadata": _context_metadata_from_state(state),
    }


def _apply_approval_decision_for_response(approval: ApprovalRequest, *, status: str, decision: str | None) -> None:
    approval.status = status
    approval.decision = decision or status
    approval.decided_at = approval.decided_at or utc_now()


def _conversation_service(session: Session) -> ConversationService:
    return ConversationService(ConversationRepository(session))


def _automation_service(session: Session) -> AutomationService:
    return AutomationService(
        workflow_runs=WorkflowRunRepository(session),
        checkpoints=WorkflowCheckpointRepository(session),
        tool_call_logs=ToolCallLogRepository(session),
        approvals=ApprovalRequestRepository(session),
    )


def _agent_graph_dependencies(session: Session, conversation_service: ConversationService) -> AgentGraphDependencies:
    automation_service = _automation_service(session)
    settings = get_settings()
    mcp_client = HttpMCPGatewayClient(server_url=settings.mcp_server_url) if settings.mcp_enabled and settings.mcp_server_url else None
    registry = create_default_agent_tool_registry(
        content_source_client=ContentSourceMCPClient(
            mcp_client=mcp_client,
            xiaohongshu_base_url=settings.xiaohongshu_mcp_base_url,
            xiaohongshu_auth_token=(
                settings.xiaohongshu_mcp_auth_token.get_secret_value() if settings.xiaohongshu_mcp_auth_token else None
            ),
        )
    )
    if settings.mcp_enabled and settings.mcp_server_url:
        registry.register_many(
            create_mcp_agent_tool_definitions(
                mcp_client,
                allowed_tool_names=settings.allowed_mcp_tools,
            )
        )
    return AgentGraphDependencies(
        automation_service=automation_service,
        checkpoint_store=AgentCheckpointStore(session=session, automation_service=automation_service),
        conversation_service=conversation_service,
        registry=registry,
        guard=AgentToolRuntimeGuard(),
        skill_repository=AgentSkillRepository(AgentMemoryRepository(session)),
        db_session=session,
        llm_client=_build_agent_llm_client(settings),
    )


def _build_agent_llm_client(settings) -> LLMChatClient | None:
    try:
        return LLMChatClient(config=build_llm_runtime_config(settings))
    except ValueError:
        return None


def _context_metadata_from_state(state) -> dict[str, object]:
    metadata = dict(getattr(state, "context_metadata", {}) or {})
    metadata.update(
        {
        "summary_id": state.latest_summary_id,
        "loaded_session_history_ids": state.loaded_session_history_ids,
        "loaded_memory_ids": state.loaded_memory_ids,
        "loaded_skill_ids": state.loaded_skill_ids,
        "token_estimate": state.token_estimate,
        "need_compaction": state.need_compaction,
        "new_user_message_included": True,
        "agent_run_id": state.agent_run_id,
        "workflow_run_id": state.workflow_run_id,
        "current_step": state.current_step,
        "tool_call_ids": state.tool_call_ids,
        "approval_request_id": state.approval_request_id,
        }
    )
    return metadata


def _sse_event(event: str, payload: dict[str, object]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"
