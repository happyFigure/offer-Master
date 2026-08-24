from __future__ import annotations

import json
from collections.abc import Iterator
from queue import Queue
from threading import Thread

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session, sessionmaker

from app.agent_runtime.checkpoints import AgentCheckpointStore
from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
from app.agent_runtime.durable_state.schemas import AgentTaskStatus
from app.agent_runtime.durable_state.service import DurableStateNotFoundError, DurableStateService
from app.agent_runtime.graph_factory import (
    AgentGraphDependencies,
    AgentPreparedResponse,
    AgentRunCommand,
    AgentWorkflowResult,
    continue_agent_workflow_after_approval,
    external_web_search_synthesis_messages,
    finalize_agent_workflow_response,
    prepare_agent_workflow_response,
    run_agent_workflow,
    tool_result_summary_response,
)
from app.agent_runtime.guardrails import AgentToolRuntimeGuard
from app.agent_runtime.loop_agent.outer_session import (
    OuterSessionLoopController,
    OuterSessionRunRequest,
    OuterSessionState,
    OuterSessionStatus,
    OuterSessionTurnResult,
)
from app.agent_runtime.loop_agent.schemas import LoopAgentAction, LoopAgentDecision, LoopAgentRunResult, LoopAgentStopReason
from app.agent_runtime.external_tasks.configured import (
    build_agent_runtime_executor_bundle,
    build_external_task_dispatcher_callback,
    build_external_web_search_callback,
)
from app.agent_runtime.memory.compaction import CompactionConfig
from app.agent_runtime.memory.skill_repository import AgentSkillRepository
from app.agent_runtime.output_sanitizer import sanitize_agent_final_answer
from app.agent_runtime.planning.execution_planner import HybridExecutionPlanner
from app.agent_runtime.routing.capability_routing_middleware import CapabilityRoutingMiddleware
from app.agent_runtime.tool_registry import create_default_agent_tool_registry, create_mcp_agent_tool_definitions
from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
from app.core.config import get_settings
from app.db.session import get_db_session
from app.domains.agent_memory.repository import AgentMemoryRepository
from app.domains.automation.repository import (
    ApprovalRequestRepository,
    ToolCallLogRepository,
    WorkflowCheckpointRepository,
    WorkflowRunRepository,
)
from app.domains.automation.models import ApprovalRequest, ToolCallLog, utc_now
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
from app.infrastructure.llm.client import build_intent_llm_runtime_config, build_llm_runtime_config
from app.mcp_gateway.content_source_client import ContentSourceMCPClient
from app.mcp_gateway.client import HttpMCPGatewayClient


router = APIRouter(prefix="/api/v1/agent", tags=["agent"])

ASSISTANT_STUB_REPLY = "我已经记录这条消息。下一步会在记忆系统接入后构建上下文。"
OUTER_SESSION_METADATA_KEY = "outer_session_loop"
STREAM_QUEUE_DONE = object()


class AgentApprovalDecisionRequest(BaseModel):
    decision_reason: str | None = None


class AgentApprovalDecisionResponse(BaseModel):
    approval: ApprovalRequestRead
    assistant_message: AgentMessageRead
    context_metadata: dict[str, object]


class AgentTaskResumeResponse(BaseModel):
    action: str
    task_id: str
    source_step_id: str | None = None
    resume_step_id: str | None = None
    reason: str | None = None
    approval_request_id: str | None = None
    executor_type: str | None = None
    executor_name: str | None = None
    capability: str | None = None
    payload: dict[str, object]
    requires_user_action: bool


@router.post("/tasks/{task_id}/resume", response_model=AgentTaskResumeResponse)
def resume_agent_task(
    task_id: str,
    session: Session = Depends(get_db_session),
) -> AgentTaskResumeResponse:
    try:
        result = DurableStateService(SqlAlchemyDurableStateRepository(session)).resume_task(task_id)
    except DurableStateNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    return AgentTaskResumeResponse(
        action=result.action.value,
        task_id=result.task_id,
        source_step_id=result.source_step_id,
        resume_step_id=result.resume_step_id,
        reason=result.reason,
        approval_request_id=result.approval_request_id,
        executor_type=result.executor_type,
        executor_name=result.executor_name,
        capability=result.capability,
        payload=result.payload,
        requires_user_action=result.requires_user_action,
    )


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
        workflow_result = _run_agent_workflow_with_outer_session(
            _agent_run_command_from_request(session_id, request),
            dependencies=dependencies,
            conversation_service=service,
        )
        assistant_content = _assistant_content_from_state(workflow_result.state)
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
        event_queue: Queue[object] = Queue()
        db_bind = session.get_bind()
        worker = Thread(
            target=_run_agent_message_stream_worker,
            kwargs={
                "session_id": session_id,
                "request": request,
                "db_bind": db_bind,
                "event_queue": event_queue,
            },
            daemon=True,
        )
        worker.start()
        while True:
            item = event_queue.get()
            if item is STREAM_QUEUE_DONE:
                break
            yield str(item)
        worker.join(timeout=1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _run_agent_message_stream_worker(
    *,
    session_id: str,
    request: AgentUserMessageRequest,
    db_bind,
    event_queue: Queue[object],
) -> None:
    SessionLocal = sessionmaker(bind=db_bind, expire_on_commit=False, future=True)
    with SessionLocal() as worker_session:
        service = _conversation_service(worker_session)
        user_message = None
        outer_request: OuterSessionRunRequest | None = None
        emitted_tool_event_count = 0
        try:
            command = _agent_run_command_from_request(session_id, request)
            outer_store = _AgentSessionOuterLoopStore(service, command.session_id)
            outer_controller = OuterSessionLoopController(
                run_inner_loop=lambda _request: LoopAgentRunResult(stop_reason=LoopAgentStopReason.STOPPED),
                store=outer_store,
            )

            def emit_tool_event(payload: dict[str, object]) -> None:
                nonlocal emitted_tool_event_count
                emitted_tool_event_count += 1
                event_queue.put(_sse_event("tool_event", payload))

            dependencies = _agent_graph_dependencies(worker_session, service).with_event_sink(emit_tool_event)

            def on_workflow_started(workflow, state) -> None:
                nonlocal user_message, outer_request
                if outer_request is not None:
                    return
                user_message, outer_request = _emit_stream_started_events(
                    service=service,
                    db_session=worker_session,
                    event_queue=event_queue,
                    session_id=session_id,
                    request=request,
                    command=command,
                    state=state,
                    workflow_run_id=workflow.id,
                    outer_store=outer_store,
                    outer_controller=outer_controller,
                    dependencies=dependencies,
                )

            prepared = prepare_agent_workflow_response(
                command,
                dependencies=dependencies,
                on_workflow_started=on_workflow_started,
            )
            if outer_request is None or user_message is None:
                user_message, outer_request = _emit_stream_started_events(
                    service=service,
                    db_session=worker_session,
                    event_queue=event_queue,
                    session_id=session_id,
                    request=request,
                    command=command,
                    state=prepared.state,
                    workflow_run_id=prepared.workflow_run_id,
                    outer_store=outer_store,
                    outer_controller=outer_controller,
                    dependencies=dependencies,
                )
            if emitted_tool_event_count == 0:
                for tool_event_payload in _tool_event_payloads_from_state(prepared.state, worker_session):
                    event_queue.put(_sse_event("tool_event", tool_event_payload))

            if prepared.state.current_step == "wait_confirmation":
                approval = worker_session.get(ApprovalRequest, prepared.state.approval_request_id)
                if approval is None:
                    raise ValueError("Agent is waiting for user confirmation, but approval request was not found.")
                turn_result = outer_controller.complete_turn(outer_request, _loop_result_from_workflow_state(prepared.state))
                _sync_outer_session_durable_turn_completed(
                    outer_request,
                    turn_result,
                    state=prepared.state,
                    dependencies=dependencies,
                    store=outer_store,
                )
                prepared_state = _state_with_outer_session_metadata(prepared.state, outer_store, turn_result)
                worker_session.commit()
                event_queue.put(
                    _sse_event(
                        "outer_session_event",
                        _outer_session_event_payload(
                            "waiting_user",
                            outer_state=outer_store.get(command.session_id),
                            turn_result=turn_result,
                        ),
                    )
                )
                event_queue.put(_sse_event("approval_required", _approval_required_payload(approval, prepared_state)))
                return

            chunks: list[str] = []
            if prepared.state.final_response and prepared.state.response_mode != "deterministic_stub":
                prepared_state, sanitized_response, response_mode = _sanitize_stream_final_response_before_tokens(
                    prepared.state,
                    final_response=prepared.state.final_response,
                    response_mode=prepared.state.response_mode,
                )
                prepared = AgentPreparedResponse(
                    workflow_run_id=prepared.workflow_run_id,
                    workflow=prepared.workflow,
                    state=prepared_state,
                )
                chunk_iterable = [sanitized_response]
            else:
                synthesis_messages = external_web_search_synthesis_messages(prepared.state)
                if synthesis_messages is not None and dependencies.llm_client is not None and hasattr(dependencies.llm_client, "stream_complete"):
                    response_mode = "llm_stream_tool_result_summary"
                    chunk_iterable = dependencies.llm_client.stream_complete(messages=synthesis_messages)
                elif synthesis_messages is not None and dependencies.llm_client is not None:
                    response_mode = "llm_tool_result_summary"
                    chunk_iterable = [dependencies.llm_client.complete(messages=synthesis_messages).content]
                else:
                    tool_response = tool_result_summary_response(prepared.state)
                    if tool_response is not None:
                        assistant_content, response_mode = tool_response
                        chunk_iterable = [assistant_content]
                    elif dependencies.llm_client is None:
                        response_mode = "deterministic_stub"
                        chunk_iterable = [ASSISTANT_STUB_REPLY]
                    elif hasattr(dependencies.llm_client, "stream_complete"):
                        response_mode = "llm_stream"
                        chunk_iterable = dependencies.llm_client.stream_complete(messages=prepared.state.llm_messages)
                    else:
                        response_mode = "llm"
                        chunk_iterable = [dependencies.llm_client.complete(messages=prepared.state.llm_messages).content]

            for chunk in chunk_iterable:
                if not chunk:
                    continue
                chunks.append(str(chunk))
                event_queue.put(_sse_event("token", {"content": str(chunk)}))

            assistant_content = "".join(chunks).strip() or ASSISTANT_STUB_REPLY
            workflow_result = finalize_agent_workflow_response(
                prepared.state,
                final_response=assistant_content,
                response_mode=response_mode,
                dependencies=dependencies,
            )
            turn_result = outer_controller.complete_turn(outer_request, _loop_result_from_workflow_state(workflow_result.state))
            _sync_outer_session_durable_turn_completed(
                outer_request,
                turn_result,
                state=workflow_result.state,
                dependencies=dependencies,
                store=outer_store,
            )
            workflow_result = AgentWorkflowResult(
                workflow_run_id=workflow_result.workflow_run_id,
                state=_state_with_outer_session_metadata(workflow_result.state, outer_store, turn_result),
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
            worker_session.commit()
            event_queue.put(
                _sse_event(
                    "outer_session_event",
                    _outer_session_event_payload(
                        "waiting_user" if turn_result.status == OuterSessionStatus.WAITING_USER else "task_finished",
                        outer_state=outer_store.get(command.session_id),
                        turn_result=turn_result,
                    ),
                )
            )
            event_queue.put(
                _sse_event(
                    "done",
                    {
                        "assistant_message": AgentMessageRead.model_validate(assistant_message).model_dump(mode="json"),
                        "context_metadata": _context_metadata_from_state(workflow_result.state),
                    },
                )
            )
        except ValueError as exc:
            worker_session.rollback()
            event_queue.put(_sse_event("error", {"message": str(exc)}))
        except Exception as exc:  # pragma: no cover - defensive streaming boundary.
            worker_session.rollback()
            event_queue.put(_sse_event("error", {"message": f"Agent stream failed: {exc}"}))
        finally:
            event_queue.put(STREAM_QUEUE_DONE)


def _emit_stream_started_events(
    *,
    service: ConversationService,
    db_session: Session,
    event_queue: Queue[object],
    session_id: str,
    request: AgentUserMessageRequest,
    command: AgentRunCommand,
    state,
    workflow_run_id: str,
    outer_store: _AgentSessionOuterLoopStore,
    outer_controller: OuterSessionLoopController,
    dependencies: AgentGraphDependencies,
):
    outer_request = outer_controller.begin_user_message(
        session_id=command.session_id,
        user_message=command.user_message,
        run_id=workflow_run_id,
    )
    _sync_outer_session_durable_turn_started(outer_request, dependencies=dependencies, store=outer_store)
    user_message = service.append_message(
        session_id,
        AgentMessageCreate(
            role=AgentMessageRole.USER,
            content_text=request.content_text,
            visible_content_text=request.content_text,
            agent_run_id=state.agent_run_id,
            workflow_run_id=workflow_run_id,
            metadata_json=request.metadata_json,
        ),
    )
    db_session.commit()
    event_queue.put(_sse_event("user_message", {"message": AgentMessageRead.model_validate(user_message).model_dump(mode="json")}))
    event_queue.put(
        _sse_event(
            "outer_session_event",
            _outer_session_event_payload(
                "task_started",
                outer_state=outer_store.get(command.session_id),
                turn_request=outer_request,
            ),
        )
    )
    return user_message, outer_request


def _sanitize_stream_final_response_before_tokens(state, *, final_response: str, response_mode: str):
    sanitized = sanitize_agent_final_answer(final_response)
    if not sanitized.removed_internal_protocol:
        return state, final_response, response_mode

    metadata = {
        **state.context_metadata,
        "output_sanitizer": {
            "removed_internal_protocol": sanitized.removed_internal_protocol,
            "needs_regeneration": sanitized.needs_regeneration,
            "removed_fragment_count": len(sanitized.removed_fragments),
            "applied_before_stream_tokens": True,
        },
    }
    state = state.with_updates(context_metadata=metadata)
    if sanitized.content:
        return state, sanitized.content, response_mode
    return (
        state,
        "我已完成处理，但最终回答需要重新整理。请重新发送问题或换一种问法。",
        "sanitized_empty_fallback",
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
        workflow_result = _complete_outer_session_after_approval(
            workflow_result,
            dependencies=dependencies,
            conversation_service=service,
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
        workflow_result = _complete_outer_session_after_approval(
            workflow_result,
            dependencies=dependencies,
            conversation_service=service,
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


def _run_agent_workflow_with_outer_session(
    command: AgentRunCommand,
    *,
    dependencies: AgentGraphDependencies,
    conversation_service: ConversationService,
) -> AgentWorkflowResult:
    store = _AgentSessionOuterLoopStore(conversation_service, command.session_id)
    outer_controller = OuterSessionLoopController(
        run_inner_loop=lambda _request: LoopAgentRunResult(stop_reason=LoopAgentStopReason.STOPPED),
        store=store,
    )
    turn_request = outer_controller.begin_user_message(
        session_id=command.session_id,
        user_message=command.user_message,
    )
    workflow_result = run_agent_workflow(command, dependencies=dependencies)
    turn_request = _outer_request_with_run_id(turn_request, workflow_result.workflow_run_id)
    _sync_outer_session_durable_turn_started(turn_request, dependencies=dependencies, store=store)
    turn_result = outer_controller.complete_turn(turn_request, _loop_result_from_workflow_state(workflow_result.state))
    _sync_outer_session_durable_turn_completed(
        turn_request,
        turn_result,
        state=workflow_result.state,
        dependencies=dependencies,
        store=store,
    )
    state = _state_with_outer_session_metadata(workflow_result.state, store, turn_result)
    return AgentWorkflowResult(workflow_run_id=workflow_result.workflow_run_id, state=state)


def _outer_request_with_run_id(request: OuterSessionRunRequest, run_id: str) -> OuterSessionRunRequest:
    if request.run_id == run_id:
        return request
    return OuterSessionRunRequest(
        session_id=request.session_id,
        task_id=request.task_id,
        run_id=run_id,
        user_goal=request.user_goal,
        user_message=request.user_message,
        is_resume=request.is_resume,
        turn_index=request.turn_index,
        resume_context=dict(request.resume_context),
    )


def _complete_outer_session_after_approval(
    workflow_result: AgentWorkflowResult,
    *,
    dependencies: AgentGraphDependencies,
    conversation_service: ConversationService,
) -> AgentWorkflowResult:
    store = _AgentSessionOuterLoopStore(conversation_service, workflow_result.state.session_id)
    outer_state = store.get(workflow_result.state.session_id)
    if outer_state is None:
        return workflow_result

    turn_request = OuterSessionRunRequest(
        session_id=workflow_result.state.session_id,
        task_id=outer_state.active_task_id,
        run_id=workflow_result.workflow_run_id,
        user_goal=outer_state.user_goal,
        user_message=workflow_result.state.user_message,
        is_resume=True,
        turn_index=max(outer_state.run_count, 1),
        resume_context={"approval_resume": True, "waiting_message": outer_state.waiting_message},
    )
    turn_result = OuterSessionLoopController(
        run_inner_loop=lambda _request: LoopAgentRunResult(stop_reason=LoopAgentStopReason.STOPPED),
        store=store,
    ).complete_turn(turn_request, _loop_result_from_workflow_state(workflow_result.state))
    _sync_outer_session_durable_turn_completed(
        turn_request,
        turn_result,
        state=workflow_result.state,
        dependencies=dependencies,
        store=store,
    )
    return AgentWorkflowResult(
        workflow_run_id=workflow_result.workflow_run_id,
        state=_state_with_outer_session_metadata(workflow_result.state, store, turn_result),
    )


def _sync_outer_session_durable_turn_started(
    turn_request: OuterSessionRunRequest,
    *,
    dependencies: AgentGraphDependencies,
    store: _AgentSessionOuterLoopStore,
) -> None:
    service = dependencies.durable_state_service
    if service is None:
        return
    step_id = _outer_durable_turn_step_id(turn_request)
    try:
        try:
            task = service.get_task(turn_request.task_id)
            task.status = AgentTaskStatus.RUNNING
            task.current_step_id = step_id
            task.user_goal = turn_request.user_goal
            task.input_payload = {
                **dict(task.input_payload or {}),
                "session_id": turn_request.session_id,
                "latest_user_message": turn_request.user_message,
                "latest_workflow_run_id": turn_request.run_id,
                "run_count": turn_request.turn_index,
                "last_is_resume": turn_request.is_resume,
            }
            service.repository.update_task(task)
        except DurableStateNotFoundError:
            service.create_task(
                task_id=turn_request.task_id,
                root_workflow_run_id=turn_request.run_id,
                conversation_session_id=turn_request.session_id,
                task_type="outer_session_task",
                capability="agent.outer_session",
                owner_executor="offermaster_outer_loop",
                user_goal=turn_request.user_goal,
                input_payload={
                    "session_id": turn_request.session_id,
                    "latest_user_message": turn_request.user_message,
                    "latest_workflow_run_id": turn_request.run_id,
                    "run_count": turn_request.turn_index,
                    "last_is_resume": turn_request.is_resume,
                },
                output_payload={"workflow_run_ids": []},
            )
        try:
            service.get_step(step_id)
        except DurableStateNotFoundError:
            service.add_step(
                task_id=turn_request.task_id,
                step_id=step_id,
                sequence_index=turn_request.turn_index,
                step_type="outer_session_turn",
                executor_type="runtime",
                executor_name="offermaster_outer_loop",
                capability="agent.outer_session",
                input_payload=turn_request.to_metadata_dict(),
            )
        service.mark_step_running(step_id)
        _annotate_outer_state_with_durable_task(
            store,
            session_id=turn_request.session_id,
            task_id=turn_request.task_id,
            workflow_run_id=turn_request.run_id,
            durable_status=AgentTaskStatus.RUNNING.value,
        )
    except Exception:
        return


def _sync_outer_session_durable_turn_completed(
    turn_request: OuterSessionRunRequest,
    turn_result: OuterSessionTurnResult,
    *,
    state,
    dependencies: AgentGraphDependencies,
    store: _AgentSessionOuterLoopStore,
) -> None:
    service = dependencies.durable_state_service
    if service is None:
        return
    step_id = _outer_durable_turn_step_id(turn_request)
    task_status = _durable_task_status_from_outer_status(turn_result.status)
    output_payload = {
        "status": turn_result.status.value,
        "workflow_run_id": state.workflow_run_id,
        "agent_run_id": state.agent_run_id,
        "response_mode": state.response_mode,
        "current_step": state.current_step,
        "requires_user_action": turn_result.requires_user_action,
        "waiting_message": turn_result.waiting_message,
        "final_answer": turn_result.final_answer,
    }
    try:
        _sync_outer_session_durable_turn_started(turn_request, dependencies=dependencies, store=store)
        if turn_result.status == OuterSessionStatus.WAITING_USER:
            service.mark_step_waiting_user(step_id, output_payload=output_payload)
        elif turn_result.status == OuterSessionStatus.FINISHED:
            service.mark_step_succeeded(step_id, output_payload=output_payload)
        else:
            service.mark_step_failed(step_id, output_payload=output_payload)

        task = service.get_task(turn_result.task_id)
        workflow_run_ids = _append_unique_strings(
            (task.output_payload or {}).get("workflow_run_ids"),
            state.workflow_run_id,
        )
        task.status = task_status
        task.current_step_id = step_id
        task.user_goal = turn_request.user_goal
        task.output_payload = {
            **dict(task.output_payload or {}),
            "status": turn_result.status.value,
            "latest_workflow_run_id": state.workflow_run_id,
            "workflow_run_ids": workflow_run_ids,
            "requires_user_action": turn_result.requires_user_action,
            "waiting_message": turn_result.waiting_message,
            "final_answer": turn_result.final_answer,
        }
        if task_status in {AgentTaskStatus.SUCCEEDED, AgentTaskStatus.FAILED}:
            task.completed_at = utc_now()
        else:
            task.completed_at = None
        service.repository.update_task(task)
        _annotate_outer_state_with_durable_task(
            store,
            session_id=turn_request.session_id,
            task_id=turn_request.task_id,
            workflow_run_id=state.workflow_run_id,
            durable_status=task_status.value,
        )
    except Exception:
        return


def _outer_durable_turn_step_id(turn_request: OuterSessionRunRequest) -> str:
    return f"{turn_request.task_id}:turn-{turn_request.turn_index}"


def _durable_task_status_from_outer_status(status: OuterSessionStatus) -> AgentTaskStatus:
    if status == OuterSessionStatus.WAITING_USER:
        return AgentTaskStatus.WAITING_USER
    if status == OuterSessionStatus.FINISHED:
        return AgentTaskStatus.SUCCEEDED
    if status == OuterSessionStatus.RUNNING:
        return AgentTaskStatus.RUNNING
    return AgentTaskStatus.FAILED


def _append_unique_strings(values: object, value: object) -> list[str]:
    result = [str(item) for item in values or [] if str(item or "").strip()]
    text = str(value or "").strip()
    if text and text not in result:
        result.append(text)
    return result


def _annotate_outer_state_with_durable_task(
    store: _AgentSessionOuterLoopStore,
    *,
    session_id: str,
    task_id: str,
    workflow_run_id: str,
    durable_status: str,
) -> None:
    outer_state = store.get(session_id)
    if outer_state is None:
        return
    outer_state.active_run_id = workflow_run_id
    outer_state.metadata = {
        **dict(outer_state.metadata),
        "durable_task_id": task_id,
        "latest_workflow_run_id": workflow_run_id,
        "durable_status": durable_status,
    }
    store.save(outer_state)


def _state_with_outer_session_metadata(state, store: _AgentSessionOuterLoopStore, turn_result: OuterSessionTurnResult):
    return state.with_updates(
        context_metadata={
            **state.context_metadata,
            OUTER_SESSION_METADATA_KEY: _outer_session_metadata_for_turn(store, state.session_id, turn_result),
        }
    )


def _outer_session_metadata_for_turn(
    store: _AgentSessionOuterLoopStore,
    session_id: str,
    turn_result: OuterSessionTurnResult,
) -> dict[str, object]:
    outer_state = store.get(session_id)
    outer_metadata = outer_state.to_metadata_dict() if outer_state is not None else {}
    outer_metadata["turn"] = {
        "status": turn_result.status.value,
        "task_id": turn_result.task_id,
        "run_id": turn_result.run_id,
        "requires_user_action": turn_result.requires_user_action,
        "waiting_message": turn_result.waiting_message,
        "final_answer": turn_result.final_answer,
    }
    return outer_metadata


def _outer_session_event_payload(
    event_type: str,
    *,
    outer_state: OuterSessionState | None,
    turn_request: OuterSessionRunRequest | None = None,
    turn_result: OuterSessionTurnResult | None = None,
) -> dict[str, object]:
    labels = {
        "task_started": "任务开始",
        "waiting_user": "等待用户补充信息",
        "task_finished": "任务结束",
    }
    summaries = {
        "task_started": "主 agent 已接收本轮用户消息，外层会话进入运行中。",
        "waiting_user": "主 agent 已暂停当前任务，正在等待用户补充信息或确认。",
        "task_finished": "主 agent 已完成当前任务，本轮流式回复结束。",
    }
    state_metadata = outer_state.to_metadata_dict() if outer_state is not None else {}
    status = turn_result.status.value if turn_result is not None else str(state_metadata.get("status") or OuterSessionStatus.RUNNING.value)
    session_id = (
        turn_result.session_id
        if turn_result is not None
        else turn_request.session_id
        if turn_request is not None
        else str(state_metadata.get("session_id") or "")
    )
    task_id = (
        turn_result.task_id
        if turn_result is not None
        else turn_request.task_id
        if turn_request is not None
        else str(state_metadata.get("active_task_id") or "")
    )
    run_id = (
        turn_result.run_id
        if turn_result is not None
        else turn_request.run_id
        if turn_request is not None
        else str(state_metadata.get("active_run_id") or "")
    )
    payload: dict[str, object] = {
        "event_type": event_type,
        "event_label": labels.get(event_type, event_type),
        "session_id": session_id,
        "task_id": task_id,
        "run_id": run_id,
        "status": status,
        "summary": summaries.get(event_type, event_type),
        "outer_session": state_metadata,
    }
    if turn_request is not None:
        payload.update(
            {
                "is_resume": turn_request.is_resume,
                "turn_index": turn_request.turn_index,
                "user_goal": turn_request.user_goal,
            }
        )
    if turn_result is not None:
        payload.update(
            {
                "requires_user_action": turn_result.requires_user_action,
                "waiting_message": turn_result.waiting_message,
                "final_answer": turn_result.final_answer,
                "turn": {
                    "status": turn_result.status.value,
                    "task_id": turn_result.task_id,
                    "run_id": turn_result.run_id,
                    "requires_user_action": turn_result.requires_user_action,
                    "waiting_message": turn_result.waiting_message,
                    "final_answer": turn_result.final_answer,
                },
            }
        )
    return payload


def _tool_event_payloads_from_state(state, db_session: Session) -> list[dict[str, object]]:
    trace_events = _tool_event_payloads_from_loop_trace(state)
    if trace_events:
        return trace_events
    return _tool_event_payloads_from_tool_logs(state, db_session)


def _tool_event_payloads_from_loop_trace(state) -> list[dict[str, object]]:
    loop_metadata = state.context_metadata.get("loop_agent") if isinstance(state.context_metadata, dict) else None
    trace = loop_metadata.get("trace") if isinstance(loop_metadata, dict) else None
    if not isinstance(trace, list):
        return []
    events: list[dict[str, object]] = []
    for fallback_index, raw_entry in enumerate(trace, start=1):
        if not isinstance(raw_entry, dict) or raw_entry.get("action") != LoopAgentAction.CALL_TOOL.value:
            continue
        step_index = _safe_int(raw_entry.get("iteration"), fallback_index)
        tool_name = str(raw_entry.get("capability") or state.requested_tool_name or "unknown_tool")
        tool_call_id = str(raw_entry.get("tool_call_id") or "") or None
        metadata = raw_entry.get("metadata") if isinstance(raw_entry.get("metadata"), dict) else {}
        tool_input_keys = [str(item) for item in metadata.get("tool_input_keys") or []]
        events.append(
            _tool_event_payload(
                "tool_started",
                state=state,
                step_index=step_index,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                status="running",
                summary=f"开始调用工具：{tool_name}。",
                tool_input_keys=tool_input_keys,
            )
        )
        events.append(
            _tool_event_payload(
                "tool_finished",
                state=state,
                step_index=step_index,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                status=str(raw_entry.get("observation_status") or "unknown"),
                summary=str(raw_entry.get("observation_summary") or f"工具执行完成：{tool_name}。"),
                tool_input_keys=tool_input_keys,
            )
        )
        reflection = metadata.get("reflection") if isinstance(metadata.get("reflection"), dict) else None
        if isinstance(reflection, dict) and reflection.get("next_action") == "retry":
            suggested_input_patch = reflection.get("suggested_input_patch") if isinstance(reflection.get("suggested_input_patch"), dict) else {}
            events.append(
                _tool_event_payload(
                    "tool_reflection_retry",
                    state=state,
                    step_index=step_index,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    status="retry",
                    summary=str(reflection.get("reason") or "工具结果不够好，准备修改输入后重试。"),
                    tool_input_keys=tool_input_keys,
                    reflection=reflection,
                    suggested_input_patch=dict(suggested_input_patch),
                )
            )
    return events


def _tool_event_payloads_from_tool_logs(state, db_session: Session) -> list[dict[str, object]]:
    tool_call_ids = [str(item) for item in getattr(state, "tool_call_ids", []) or [] if str(item)]
    if not tool_call_ids:
        return []
    events: list[dict[str, object]] = []
    for step_index, tool_call_id in enumerate(tool_call_ids, start=1):
        tool_log = db_session.get(ToolCallLog, tool_call_id)
        tool_name = str(getattr(tool_log, "tool_name", None) or state.requested_tool_name or "unknown_tool")
        input_payload = tool_log.input_payload if tool_log is not None and isinstance(tool_log.input_payload, dict) else {}
        output_payload = tool_log.output_payload if tool_log is not None and isinstance(tool_log.output_payload, dict) else {}
        status = _enum_value(getattr(tool_log, "status", None) or "unknown")
        error = str(getattr(tool_log, "error", None) or "") or None
        events.append(
            _tool_event_payload(
                "tool_started",
                state=state,
                step_index=step_index,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                status="running",
                summary=f"开始调用工具：{tool_name}。",
                tool_input_keys=sorted(str(key) for key in input_payload.keys()),
            )
        )
        events.append(
            _tool_event_payload(
                "tool_finished",
                state=state,
                step_index=step_index,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                status=status,
                summary=_tool_log_summary(tool_name, status=status, output_payload=output_payload, error=error),
                tool_input_keys=sorted(str(key) for key in input_payload.keys()),
            )
        )
    return events


def _tool_event_payload(
    event_type: str,
    *,
    state,
    step_index: int,
    tool_name: str,
    tool_call_id: str | None,
    status: str,
    summary: str,
    tool_input_keys: list[str],
    reflection: dict[str, object] | None = None,
    suggested_input_patch: dict[str, object] | None = None,
) -> dict[str, object]:
    labels = {
        "tool_started": "工具开始",
        "tool_finished": "工具完成",
        "tool_reflection_retry": "准备重试",
    }
    payload: dict[str, object] = {
        "event_type": event_type,
        "event_label": labels.get(event_type, event_type),
        "session_id": state.session_id,
        "workflow_run_id": state.workflow_run_id,
        "agent_run_id": state.agent_run_id,
        "step_index": step_index,
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "status": status,
        "summary": summary,
        "tool_input_keys": list(tool_input_keys),
    }
    if reflection is not None:
        payload["reflection"] = dict(reflection)
    if suggested_input_patch is not None:
        payload["suggested_input_patch"] = dict(suggested_input_patch)
    return payload


def _tool_log_summary(tool_name: str, *, status: str, output_payload: dict[str, object], error: str | None) -> str:
    if error:
        return error
    if tool_name == "external.web_search":
        return f"工具执行完成：{tool_name}，状态：{status}。"
    summary = _summary_from_output_payload(output_payload)
    if summary:
        return summary
    return f"工具执行完成：{tool_name}，状态：{status}。"


def _summary_from_output_payload(output_payload: dict[str, object]) -> str | None:
    result_payload = output_payload.get("result") if isinstance(output_payload.get("result"), dict) else output_payload
    if not isinstance(result_payload, dict):
        return None
    envelope = result_payload.get("result_envelope") if isinstance(result_payload.get("result_envelope"), dict) else None
    if envelope is None:
        nested_result = result_payload.get("result") if isinstance(result_payload.get("result"), dict) else None
        envelope = nested_result.get("result_envelope") if isinstance(nested_result, dict) and isinstance(nested_result.get("result_envelope"), dict) else None
    if isinstance(envelope, dict) and envelope.get("summary"):
        return str(envelope["summary"])
    nested = result_payload.get("result") if isinstance(result_payload.get("result"), dict) else {}
    for candidate in (nested.get("status"), result_payload.get("status")):
        if candidate:
            text = str(candidate).strip()
            return text[:500] if len(text) > 500 else text
    return None


def _enum_value(value: object) -> str:
    raw_value = getattr(value, "value", value)
    return str(raw_value or "") or "unknown"


def _safe_int(value: object, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


class _AgentSessionOuterLoopStore:
    def __init__(self, conversation_service: ConversationService, session_id: str) -> None:
        self._conversation_service = conversation_service
        self._session_id = session_id

    def get(self, session_id: str) -> OuterSessionState | None:
        if session_id != self._session_id:
            return None
        agent_session = self._conversation_service.get_session(session_id)
        metadata = agent_session.metadata_json if isinstance(agent_session.metadata_json, dict) else {}
        return _outer_session_state_from_metadata(metadata.get(OUTER_SESSION_METADATA_KEY))

    def save(self, state: OuterSessionState) -> OuterSessionState:
        agent_session = self._conversation_service.get_session(state.session_id)
        metadata = dict(agent_session.metadata_json or {})
        metadata[OUTER_SESSION_METADATA_KEY] = state.to_metadata_dict()
        agent_session.metadata_json = metadata
        return state


def _loop_result_from_workflow_state(state) -> LoopAgentRunResult:
    if _state_requires_outer_user_action(state):
        message = _outer_waiting_message_from_state(state)
        return LoopAgentRunResult(
            stop_reason=LoopAgentStopReason.WAITING_USER,
            requires_user_action=True,
            pending_decision=LoopAgentDecision(
                action=LoopAgentAction.WAIT_USER,
                message=message,
                reason=message,
            ),
            metadata={"workflow_run_id": state.workflow_run_id, "response_mode": state.response_mode},
        )
    return LoopAgentRunResult(
        stop_reason=LoopAgentStopReason.MODEL_FINAL,
        final_answer=_assistant_content_from_state(state),
        metadata={"workflow_run_id": state.workflow_run_id, "response_mode": state.response_mode},
    )


def _state_requires_outer_user_action(state) -> bool:
    if state.current_step == "wait_confirmation":
        return True
    if state.response_mode in {"clarification_ask_user", "capability_route_ask_user", "execution_planner_ask_user"}:
        return True
    loop_metadata = state.context_metadata.get("loop_agent") if isinstance(state.context_metadata, dict) else None
    return bool(
        isinstance(loop_metadata, dict)
        and (
            loop_metadata.get("requires_user_action")
            or loop_metadata.get("stop_reason") == LoopAgentStopReason.WAITING_USER.value
        )
    )


def _outer_waiting_message_from_state(state) -> str:
    if state.final_response:
        return state.final_response
    guard_result = state.guard_result if isinstance(state.guard_result, dict) else {}
    for key in ("user_message", "reason", "message"):
        value = guard_result.get(key)
        if value:
            return str(value)
    return "我需要你补充信息后才能继续。"


def _outer_session_state_from_metadata(payload: object) -> OuterSessionState | None:
    if not isinstance(payload, dict):
        return None
    try:
        status = OuterSessionStatus(str(payload.get("status") or OuterSessionStatus.RUNNING.value))
    except ValueError:
        status = OuterSessionStatus.RUNNING
    session_id = str(payload.get("session_id") or "").strip()
    active_task_id = str(payload.get("active_task_id") or "").strip()
    user_goal = str(payload.get("user_goal") or "").strip()
    if not session_id or not active_task_id or not user_goal:
        return None
    return OuterSessionState(
        session_id=session_id,
        active_task_id=active_task_id,
        user_goal=user_goal,
        status=status,
        run_count=int(payload.get("run_count") or 0),
        active_run_id=str(payload.get("active_run_id") or "") or None,
        waiting_message=str(payload.get("waiting_message") or "") or None,
        pending_decision=_loop_decision_from_metadata(payload.get("pending_decision")),
        user_followups=[str(item) for item in payload.get("user_followups") or []],
        metadata=dict(payload.get("metadata") or {}),
    )


def _loop_decision_from_metadata(payload: object) -> LoopAgentDecision | None:
    if not isinstance(payload, dict):
        return None
    try:
        action = LoopAgentAction(str(payload.get("action") or LoopAgentAction.WAIT_USER.value))
    except ValueError:
        action = LoopAgentAction.WAIT_USER
    tool_input = payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return LoopAgentDecision(
        action=action,
        capability=str(payload.get("capability") or "") or None,
        tool_input=dict(tool_input),
        message=str(payload.get("message") or "") or None,
        reason=str(payload.get("reason") or "") or None,
        metadata=dict(metadata),
    )


def _append_assistant_message_from_state(service: ConversationService, state) -> object:
    assistant_content = _assistant_content_from_state(state)
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


def _assistant_content_from_state(state) -> str:
    if state.final_response and state.response_mode != "deterministic_stub":
        return state.final_response
    return ASSISTANT_STUB_REPLY


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
        ),
        external_task_dispatcher=build_external_task_dispatcher_callback(settings),
        external_web_search_executor=build_external_web_search_callback(settings),
    )
    if settings.mcp_enabled and settings.mcp_server_url:
        registry.register_many(
            create_mcp_agent_tool_definitions(
                mcp_client,
                allowed_tool_names=settings.allowed_mcp_tools,
            )
        )
    agent_executors, capability_executor_ids = build_agent_runtime_executor_bundle(settings)
    return AgentGraphDependencies(
        automation_service=automation_service,
        checkpoint_store=AgentCheckpointStore(session=session, automation_service=automation_service),
        conversation_service=conversation_service,
        registry=registry,
        guard=AgentToolRuntimeGuard(),
        skill_repository=AgentSkillRepository(AgentMemoryRepository(session)),
        db_session=session,
        llm_client=_build_agent_llm_client(settings),
        intent_detector=_build_agent_intent_detector(settings),
        execution_planner=_build_agent_execution_planner(settings),
        capability_routing_middleware=CapabilityRoutingMiddleware(),
        durable_state_service=DurableStateService(SqlAlchemyDurableStateRepository(session)),
        agent_executors=agent_executors,
        capability_executor_ids=capability_executor_ids,
    )


def _build_agent_llm_client(settings) -> LLMChatClient | None:
    try:
        return LLMChatClient(config=build_llm_runtime_config(settings))
    except ValueError:
        return None


def _build_agent_intent_detector(settings) -> HybridIntentDetector:
    try:
        intent_llm_client = LLMChatClient(config=build_intent_llm_runtime_config(settings))
    except ValueError:
        intent_llm_client = None
    return HybridIntentDetector(llm_client=intent_llm_client)


def _build_agent_execution_planner(settings) -> HybridExecutionPlanner | None:
    if not settings.execution_planner_enabled:
        return None
    try:
        planner_llm_client = LLMChatClient(config=build_llm_runtime_config(settings))
    except ValueError:
        return None
    return HybridExecutionPlanner(llm_client=planner_llm_client)


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
