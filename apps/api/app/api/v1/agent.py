from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime
from queue import Queue
from threading import Thread

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.agent_runtime.checkpoints import AgentCheckpointStore
from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
from app.agent_runtime.durable_state.models import AgentStepState, AgentTaskState
from app.agent_runtime.durable_state.schemas import AgentStepStatus, AgentTaskStatus
from app.agent_runtime.durable_state.service import DurableStateNotFoundError, DurableStateService
from app.agent_runtime.graph_factory import (
    AgentGraphDependencies,
    AgentPreparedResponse,
    AgentRunCommand,
    AgentWorkflowResult,
    LOOP_RUNNER_STAGE_CONTEXT_METADATA_KEY,
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
from app.agent_runtime.output_sanitizer import (
    contains_false_tool_execution_claim,
    could_be_false_tool_execution_claim_prefix,
    false_tool_execution_claim_fallback_response,
    sanitize_agent_final_answer,
)
from app.agent_runtime.planning.execution_planner import HybridExecutionPlanner
from app.agent_runtime.routing.capability_routing_middleware import CapabilityRoutingMiddleware
from app.agent_runtime.tool_registry import (
    EXTERNAL_WEB_SEARCH_TOOL,
    LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,
    LOCAL_JOB_SOURCE_OVERVIEW_TOOL,
    create_default_agent_tool_registry,
    create_mcp_agent_tool_definitions,
)
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
DEFAULT_TASK_PLAN_STAGES: tuple[dict[str, object], ...] = (
    {
        "stage_id": "clarify_goal",
        "title": "明确目标和约束",
        "objective": "确认用户要完成什么、有什么范围限制、最终需要什么输出。",
        "business_action": "先读懂用户目标、数量、范围和输出形式；如果本轮候选工具已经非常明确，可以交给模型选择工具。",
        "tool_strategy": {
            "mode": "inherit",
            "description": "继承本轮候选工具，不额外限制，避免一开始就漏掉用户真正需要的工具。",
        },
        "capability": "agent.stage.clarify_goal",
        "depends_on": [],
    },
    {
        "stage_id": "collect_candidates",
        "title": "收集本地候选信息",
        "objective": "查询本地公司库、岗位线索、校招来源等已有信息。",
        "business_action": "先看 OfferMaster 本地数据库里已经有什么公司、岗位线索和校招来源。",
        "allowed_capabilities": [LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL, LOCAL_JOB_SOURCE_OVERVIEW_TOOL],
        "tool_strategy": {
            "mode": "allowlist",
            "description": "本阶段只允许本地只读概览工具，不直接联网，也不做会修改数据的同步动作。",
        },
        "capability": "agent.stage.collect_candidates",
        "depends_on": ["clarify_goal"],
    },
    {
        "stage_id": "enrich_external_info",
        "title": "补充外部公开信息",
        "objective": "必要时联网补充公司主营业务、招聘动态、公开资料来源。",
        "business_action": "基于上一步的候选公司或用户指定对象，补充公开网页、公众号、小红书等外部资料。",
        "allowed_capabilities": [
            EXTERNAL_WEB_SEARCH_TOOL,
            "weixin-articles-mcp.read_article",
            "xiaohongshu-mcp.search_feeds",
            "xiaohongshu-mcp.get_feed_detail",
        ],
        "tool_strategy": {
            "mode": "allowlist",
            "description": "本阶段只允许公开信息读取和搜索类工具，用来补全外部资料。",
        },
        "capability": "agent.stage.enrich_external_info",
        "depends_on": ["collect_candidates"],
    },
    {
        "stage_id": "analyze_rank",
        "title": "分析匹配和排序",
        "objective": "结合用户目标、岗位要求和候选信息，对结果做匹配分析和排序。",
        "business_action": "把已有本地信息和外部补充信息放在一起做判断、筛选、排序和理由说明。",
        "ranking_policy": [
            "技术匹配度：优先岗位方向、技术栈和用户目标一致的公司。",
            "城市和地点：优先用户明确要求或更适合用户投递的城市。",
            "校招确定性：优先有明确校招、岗位、官网或可信公开证据的公司。",
            "投递优先级：优先信息完整、岗位质量高、申请路径清晰的机会。",
        ],
        "tool_strategy": {
            "mode": "none",
            "description": "本阶段不再扩展检索范围，优先基于已有阶段产物做分析。",
        },
        "capability": "agent.stage.analyze_rank",
        "depends_on": ["enrich_external_info"],
    },
    {
        "stage_id": "finalize_answer",
        "title": "整理最终输出",
        "objective": "把执行结果整理成用户能直接理解的答案、表格或建议。",
        "business_action": "把结论整理成用户能直接使用的回答、表格、分组或下一步建议。",
        "tool_strategy": {
            "mode": "none",
            "description": "本阶段只负责表达和整理，不再调用工具。",
        },
        "capability": "agent.stage.finalize_answer",
        "depends_on": ["analyze_rank"],
    },
)


class AgentApprovalDecisionRequest(BaseModel):
    decision_reason: str | None = None


class AgentApprovalDecisionResponse(BaseModel):
    approval: ApprovalRequestRead
    assistant_message: AgentMessageRead
    context_metadata: dict[str, object]


class AgentTaskStepRead(BaseModel):
    id: str
    task_id: str
    parent_step_id: str | None
    sequence_index: int
    step_type: str
    status: str
    executor_type: str
    executor_name: str
    capability: str
    input_payload: dict[str, object]
    output_payload: dict[str, object]
    tool_call_log_id: str | None
    external_task_id: str | None
    approval_request_id: str | None
    retry_count: int
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class AgentTaskRead(BaseModel):
    id: str
    root_workflow_run_id: str
    conversation_session_id: str
    task_type: str
    capability: str
    status: str
    current_step_id: str | None
    owner_executor: str | None
    user_goal: str | None
    input_payload: dict[str, object]
    output_payload: dict[str, object]
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    steps: list[AgentTaskStepRead] = Field(default_factory=list)


class AgentTaskListResponse(BaseModel):
    items: list[AgentTaskRead]


class AgentTaskResumeResponse(BaseModel):
    action: str
    task_id: str
    source_step_id: str | None = None
    resume_step_id: str | None = None
    resume_stage_id: str | None = None
    resume_stage_title: str | None = None
    resume_stage_step_id: str | None = None
    resume_stage_status: str | None = None
    resume_stage_capability: str | None = None
    reason: str | None = None
    approval_request_id: str | None = None
    executor_type: str | None = None
    executor_name: str | None = None
    capability: str | None = None
    payload: dict[str, object]
    requires_user_action: bool
    task: AgentTaskRead | None = None


class AgentTaskRecoverRunResponse(BaseModel):
    resume: AgentTaskResumeResponse
    executed: bool
    assistant_message: AgentMessageRead | None = None
    task: AgentTaskRead
    context_metadata: dict[str, object] = Field(default_factory=dict)


class AgentTaskFollowupRequest(BaseModel):
    content_text: str = Field(min_length=1)


class AgentTaskFollowupResponse(BaseModel):
    task_id: str
    queued_count: int
    user_followups: list[str]
    task: AgentTaskRead


class AgentTaskPlanStageRead(BaseModel):
    step_id: str
    stage_id: str
    sequence_index: int
    title: str
    objective: str
    business_action: str | None = None
    allowed_capabilities: list[str] = Field(default_factory=list)
    tool_strategy: dict[str, object] = Field(default_factory=dict)
    ranking_policy: list[str] = Field(default_factory=list)
    capability: str
    status: str
    execution_status: str | None = None
    waiting_message: str | None = None
    final_answer_preview: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    received_context: dict[str, object] | None = None
    handoff_payload: dict[str, object] | None = None


class AgentTaskPlanResponse(BaseModel):
    task_id: str
    user_goal: str | None
    current_stage_id: str | None
    stages: list[AgentTaskPlanStageRead]


@router.post("/tasks/{task_id}/resume", response_model=AgentTaskResumeResponse)
def resume_agent_task(
    task_id: str,
    session: Session = Depends(get_db_session),
) -> AgentTaskResumeResponse:
    return _resume_agent_task_response(task_id, session=session)


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


@router.post("/sessions/{session_id}/tasks/recover", response_model=AgentTaskResumeResponse)
def recover_agent_session_latest_task(
    session_id: str,
    session: Session = Depends(get_db_session),
) -> AgentTaskResumeResponse:
    try:
        _conversation_service(session).get_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    task = _latest_agent_session_task(session, session_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"No agent task state found for session: {session_id}")
    return _resume_agent_task_response(task.id, session=session)


@router.post("/sessions/{session_id}/tasks/recover/run", response_model=AgentTaskRecoverRunResponse)
def recover_and_run_agent_session_latest_task(
    session_id: str,
    session: Session = Depends(get_db_session),
) -> AgentTaskRecoverRunResponse:
    service = _conversation_service(session)
    try:
        service.get_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    task = _latest_agent_session_task(session, session_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"No agent task state found for session: {session_id}")
    return _recover_and_run_agent_task_response(task.id, session=session, conversation_service=service)


@router.post("/sessions/{session_id}/tasks/followups", response_model=AgentTaskFollowupResponse)
def enqueue_agent_session_task_followup(
    session_id: str,
    request: AgentTaskFollowupRequest,
    session: Session = Depends(get_db_session),
) -> AgentTaskFollowupResponse:
    service = _conversation_service(session)
    try:
        service.get_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    task = _latest_agent_session_task(session, session_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"No agent task state found for session: {session_id}")
    return _enqueue_agent_session_task_followup(
        task,
        request=request,
        session=session,
        conversation_service=service,
    )


@router.get("/sessions/{session_id}/tasks", response_model=AgentTaskListResponse)
def list_agent_session_tasks(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db_session),
) -> AgentTaskListResponse:
    try:
        _conversation_service(session).get_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    statement = (
        select(AgentTaskState)
        .where(AgentTaskState.conversation_session_id == session_id)
        .where(AgentTaskState.task_type == "outer_session_task")
        .order_by(AgentTaskState.updated_at.desc(), AgentTaskState.created_at.desc(), AgentTaskState.id.desc())
        .offset(offset)
        .limit(limit)
    )
    tasks = list(session.scalars(statement).all())
    return AgentTaskListResponse(items=[_agent_task_read(task) for task in tasks])


@router.get("/tasks/{task_id}", response_model=AgentTaskRead)
def get_agent_task(
    task_id: str,
    session: Session = Depends(get_db_session),
) -> AgentTaskRead:
    task = session.get(AgentTaskState, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Agent task state not found: {task_id}")
    steps = list(
        session.scalars(
            select(AgentStepState)
            .where(AgentStepState.task_id == task_id)
            .order_by(AgentStepState.sequence_index.asc(), AgentStepState.created_at.asc(), AgentStepState.id.asc())
        ).all()
    )
    return _agent_task_read(task, steps=steps)


@router.get("/tasks/{task_id}/plan", response_model=AgentTaskPlanResponse)
def get_agent_task_plan(
    task_id: str,
    session: Session = Depends(get_db_session),
) -> AgentTaskPlanResponse:
    task = session.get(AgentTaskState, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Agent task state not found: {task_id}")
    return _agent_task_plan_response(task, session=session)


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
            outer_request = outer_controller.begin_user_message(
                session_id=command.session_id,
                user_message=command.user_message,
            )
            _sync_outer_session_durable_turn_started(outer_request, dependencies=dependencies, store=outer_store)

            def on_workflow_started(workflow, state):
                nonlocal user_message, outer_request
                outer_request = _outer_request_with_run_id(outer_request, workflow.id)
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
                    outer_request=outer_request,
                )
                return _state_with_loop_runner_stage_context(state, outer_request, dependencies=dependencies)

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
                    outer_request=_outer_request_with_run_id(outer_request, prepared.workflow_run_id) if outer_request is not None else None,
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
                streaming_synthesis = _prepared_tool_final_stream_synthesis(prepared_state)
                if (
                    streaming_synthesis is not None
                    and dependencies.llm_client is not None
                    and hasattr(dependencies.llm_client, "stream_complete")
                ):
                    synthesis_messages, response_mode = streaming_synthesis
                    chunk_iterable = dependencies.llm_client.stream_complete(messages=synthesis_messages)
                else:
                    chunk_iterable = _iter_visible_response_chunks(sanitized_response)
            else:
                synthesis_messages = external_web_search_synthesis_messages(prepared.state)
                if synthesis_messages is not None and dependencies.llm_client is not None and hasattr(dependencies.llm_client, "stream_complete"):
                    response_mode = "llm_stream_tool_result_summary"
                    chunk_iterable = dependencies.llm_client.stream_complete(messages=synthesis_messages)
                elif synthesis_messages is not None and dependencies.llm_client is not None:
                    response_mode = "llm_tool_result_summary"
                    chunk_iterable = _iter_visible_response_chunks(dependencies.llm_client.complete(messages=synthesis_messages).content)
                else:
                    tool_response = tool_result_summary_response(prepared.state, dependencies=dependencies)
                    if tool_response is not None:
                        assistant_content, response_mode = tool_response
                        chunk_iterable = _iter_visible_response_chunks(assistant_content)
                    elif dependencies.llm_client is None:
                        response_mode = "deterministic_stub"
                        chunk_iterable = _iter_visible_response_chunks(ASSISTANT_STUB_REPLY)
                    elif hasattr(dependencies.llm_client, "stream_complete"):
                        response_mode = "llm_stream"
                        chunk_iterable = dependencies.llm_client.stream_complete(messages=prepared.state.llm_messages)
                    else:
                        response_mode = "llm"
                        chunk_iterable = _iter_visible_response_chunks(dependencies.llm_client.complete(messages=prepared.state.llm_messages).content)

            suppressed_internal_protocol = _emit_visible_token_chunks(event_queue, chunk_iterable, chunks, state=prepared.state)

            raw_assistant_content = "".join(chunks).strip() or ASSISTANT_STUB_REPLY
            workflow_result = finalize_agent_workflow_response(
                prepared.state,
                final_response=raw_assistant_content,
                response_mode=response_mode,
                dependencies=dependencies,
            )
            if workflow_result.state.current_step == "wait_confirmation":
                approval = worker_session.get(ApprovalRequest, workflow_result.state.approval_request_id)
                if approval is None:
                    raise ValueError("Agent is waiting for user confirmation, but approval request was not found.")
                turn_result = outer_controller.complete_turn(outer_request, _loop_result_from_workflow_state(workflow_result.state))
                _sync_outer_session_durable_turn_completed(
                    outer_request,
                    turn_result,
                    state=workflow_result.state,
                    dependencies=dependencies,
                    store=outer_store,
                )
                prepared_state = _state_with_outer_session_metadata(workflow_result.state, outer_store, turn_result)
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
            assistant_content = workflow_result.state.final_response or raw_assistant_content
            if suppressed_internal_protocol and assistant_content:
                event_queue.put(_sse_event("token", {"content": assistant_content}))
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
    outer_request: OuterSessionRunRequest | None = None,
):
    if outer_request is None:
        outer_request = outer_controller.begin_user_message(
            session_id=command.session_id,
            user_message=command.user_message,
            run_id=workflow_run_id,
        )
    else:
        outer_request = _outer_request_with_run_id(outer_request, workflow_run_id)
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


def _emit_visible_token_chunks(event_queue: Queue[object], chunk_iterable, chunks: list[str], *, state) -> bool:
    """Stream normal text, but suppress internal tool protocol text before it reaches the UI."""

    probing_buffer = ""
    passthrough = False
    suppressed_internal_protocol = False
    emitted_blocked_protocol_event = False
    for chunk in chunk_iterable:
        if not chunk:
            continue
        text = str(chunk)
        chunks.append(text)
        if suppressed_internal_protocol:
            continue
        if passthrough:
            event_queue.put(_sse_event("token", {"content": text}))
            continue

        probing_buffer += text
        sanitized_probe = sanitize_agent_final_answer(probing_buffer)
        if sanitized_probe.removed_internal_protocol:
            suppressed_internal_protocol = True
            if not emitted_blocked_protocol_event:
                event_queue.put(_sse_event("tool_event", _textual_tool_call_blocked_payload(state, probing_buffer)))
                emitted_blocked_protocol_event = True
            probing_buffer = ""
            continue
        if not getattr(state, "tool_call_ids", []) and contains_false_tool_execution_claim(probing_buffer):
            suppressed_internal_protocol = True
            probing_buffer = ""
            continue
        if _could_be_internal_tool_protocol_prefix(probing_buffer) or _could_be_false_tool_execution_claim_prefix_for_state(state, probing_buffer):
            continue

        passthrough = True
        event_queue.put(_sse_event("token", {"content": probing_buffer}))
        probing_buffer = ""

    if probing_buffer and not suppressed_internal_protocol:
        event_queue.put(_sse_event("token", {"content": probing_buffer}))
    return suppressed_internal_protocol


def _iter_visible_response_chunks(content: str, *, max_chars: int = 72) -> Iterator[str]:
    """Split already-prepared text into SSE-friendly chunks without changing content."""

    text = str(content or "")
    if not text:
        return
    buffer: list[str] = []
    for char in text:
        buffer.append(char)
        joined = "".join(buffer)
        if len(joined) >= max_chars or char in "。！？!?；;\n":
            yield joined
            buffer.clear()
    if buffer:
        yield "".join(buffer)


def _prepared_tool_final_stream_synthesis(state) -> tuple[list[dict[str, object]], str] | None:
    """Build streaming final-answer messages for tool-loop states with completed tool calls."""

    if not getattr(state, "tool_call_ids", []):
        return None
    if str(getattr(state, "response_mode", "") or "") != "llm_tool_choice_loop":
        return None

    search_messages = external_web_search_synthesis_messages(state)
    if search_messages is not None:
        return search_messages, "llm_stream_tool_result_summary"

    transcript = _tool_transcript_for_final_stream_synthesis(state)
    if not transcript:
        return None

    instruction = (
        "你是 OfferMaster 的主 Agent。工具或子 agent 已经执行完，"
        "现在需要你基于用户问题和工具返回结果整理最终回答。"
        "不要把内部工具协议、Tool call、Tool result、JSON 原样展示给用户；"
        "只输出用户真正需要看的结论、关键信息和必要提醒。"
        "如果工具结果不足、失败或没有回答原问题，要直接说明不足，"
        "不要假装已经拿到了不存在的信息。用中文回答，表达简洁。"
    )
    evidence = {
        "user_message": getattr(state, "user_message", ""),
        "requested_tool_name": getattr(state, "requested_tool_name", None),
        "tool_call_ids": list(getattr(state, "tool_call_ids", []) or []),
        "tool_transcript": transcript,
    }
    evidence_message = (
        f"用户原始问题：{getattr(state, 'user_message', '')}\n\n"
        "工具执行记录 JSON：\n"
        f"{_json_for_prompt(evidence)}\n\n"
        "请根据这些工具结果生成最终回答。"
    )
    return (
        [
            {"role": "system", "content": instruction, "metadata": {"source": "tool_loop_final_synthesis"}},
            {"role": "user", "content": evidence_message, "metadata": {"source": "tool_loop_final_synthesis"}},
        ],
        "llm_stream_tool_choice_loop_final",
    )


def _tool_transcript_for_final_stream_synthesis(state) -> list[dict[str, object]]:
    transcript: list[dict[str, object]] = []
    for message in list(getattr(state, "llm_messages", []) or []):
        if not isinstance(message, dict):
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        content_json = metadata.get("content_json") if isinstance(metadata, dict) else None
        if isinstance(content_json, dict) and content_json.get("tool_name"):
            transcript.append(_trim_tool_transcript_entry(content_json))
            continue
        content = str(message.get("content") or "")
        if "tool" not in content.lower() and "工具" not in content:
            continue
        transcript.append({"content": _truncate_prompt_text(content)})
    return transcript[-12:]


def _trim_tool_transcript_entry(entry: dict[str, object]) -> dict[str, object]:
    trimmed: dict[str, object] = {}
    for key in ("tool_name", "input", "status", "result", "error"):
        if key not in entry:
            continue
        value = entry.get(key)
        if isinstance(value, str):
            trimmed[key] = _truncate_prompt_text(value)
        else:
            trimmed[key] = value
    return trimmed


def _json_for_prompt(payload: object, *, max_chars: int = 12000) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return _truncate_prompt_text(text, max_chars=max_chars)


def _truncate_prompt_text(text: str, *, max_chars: int = 12000) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}...（内容过长，已截断）"


_TEXTUAL_TOOL_CALL_NAME_RE = re.compile(
    r"(?:Tool\s*call|工具调用)\s*[:：]\s*(?P<tool>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)",
    re.IGNORECASE,
)


def _textual_tool_call_blocked_payload(state, content: str) -> dict[str, object]:
    match = _TEXTUAL_TOOL_CALL_NAME_RE.search(str(content or ""))
    tool_name = match.group("tool") if match else "unknown_tool"
    return {
        "event_type": "textual_tool_call_blocked",
        "event_label": "疑似工具调用",
        "session_id": state.session_id,
        "workflow_run_id": state.workflow_run_id,
        "agent_run_id": state.agent_run_id,
        "step_index": 1,
        "tool_name": tool_name,
        "tool_call_id": None,
        "status": "not_executed",
        "summary": "模型把工具调用写成了普通文字；这不是结构化工具调用，运行时已拦截，没有当作真实工具执行。",
        "tool_input_keys": [],
    }


def _could_be_internal_tool_protocol_prefix(content: str) -> bool:
    stripped = str(content or "").replace("\r\n", "\n").lstrip()
    if not stripped:
        return True
    normalized = stripped.lower().replace("\\_", "_")
    protocol_prefixes = (
        "tool call",
        "tool result",
        "工具调用",
        "工具结果",
        "**offermaster ai",
        "offermaster ai",
    )
    if any(prefix.startswith(normalized) for prefix in protocol_prefixes):
        return True
    assistant_label = re.match(r"(?is)^\s*(?:\*\*)?offermaster\s+ai(?:\*\*)?\s*$", stripped)
    return assistant_label is not None


def _could_be_false_tool_execution_claim_prefix_for_state(state, content: str) -> bool:
    if getattr(state, "tool_call_ids", []):
        return False
    return could_be_false_tool_execution_claim_prefix(content)


def _sanitize_stream_final_response_before_tokens(state, *, final_response: str, response_mode: str):
    sanitized = sanitize_agent_final_answer(final_response)
    if not sanitized.removed_internal_protocol:
        if not getattr(state, "tool_call_ids", []) and contains_false_tool_execution_claim(final_response):
            metadata = {
                **state.context_metadata,
                "output_sanitizer": {
                    "removed_false_tool_claim": True,
                    "needs_regeneration": True,
                    "applied_before_stream_tokens": True,
                },
            }
            return state.with_updates(context_metadata=metadata), false_tool_execution_claim_fallback_response(), "false_tool_claim_fallback"
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
        if not getattr(state, "tool_call_ids", []) and contains_false_tool_execution_claim(sanitized.content):
            metadata = {
                **state.context_metadata,
                "output_sanitizer": {
                    **dict(state.context_metadata.get("output_sanitizer") or {}),
                    "removed_false_tool_claim": True,
                    "needs_regeneration": True,
                    "applied_before_stream_tokens": True,
                },
            }
            return state.with_updates(context_metadata=metadata), false_tool_execution_claim_fallback_response(), "false_tool_claim_fallback"
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
    task_id: str | None = None,
) -> AgentWorkflowResult:
    store = _AgentSessionOuterLoopStore(conversation_service, command.session_id)
    outer_controller = OuterSessionLoopController(
        run_inner_loop=lambda _request: LoopAgentRunResult(stop_reason=LoopAgentStopReason.STOPPED),
        store=store,
    )
    turn_request = outer_controller.begin_user_message(
        session_id=command.session_id,
        user_message=command.user_message,
        task_id=task_id,
    )
    _sync_outer_session_durable_turn_started(turn_request, dependencies=dependencies, store=store)

    def on_workflow_started(workflow, state):
        return _state_with_loop_runner_stage_context(state, turn_request, dependencies=dependencies)

    workflow_result = _run_agent_workflow_with_optional_start_callback(
        command,
        dependencies=dependencies,
        on_workflow_started=on_workflow_started,
    )
    turn_request = _outer_request_with_run_id(turn_request, workflow_result.workflow_run_id)
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


def _run_agent_workflow_with_optional_start_callback(
    command: AgentRunCommand,
    *,
    dependencies: AgentGraphDependencies,
    on_workflow_started,
) -> AgentWorkflowResult:
    try:
        return run_agent_workflow(
            command,
            dependencies=dependencies,
            on_workflow_started=on_workflow_started,
        )
    except TypeError as exc:
        if "on_workflow_started" not in str(exc):
            raise
        return run_agent_workflow(command, dependencies=dependencies)


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


def _state_with_loop_runner_stage_context(state, turn_request: OuterSessionRunRequest, *, dependencies: AgentGraphDependencies):
    service = dependencies.durable_state_service
    if service is None:
        return state
    stage_context = _outer_task_stage_context_for_runner(turn_request, service=service)
    if not stage_context:
        return state
    return state.with_updates(
        context_metadata={
            **dict(state.context_metadata or {}),
            LOOP_RUNNER_STAGE_CONTEXT_METADATA_KEY: stage_context,
        }
    )


def _outer_task_stage_context_for_runner(
    turn_request: OuterSessionRunRequest,
    *,
    service: DurableStateService,
) -> dict[str, object]:
    stage_steps = [
        step
        for index in range(1, len(DEFAULT_TASK_PLAN_STAGES) + 1)
        if (step := _get_outer_task_stage_step(turn_request.task_id, index, service=service)) is not None
    ]
    current_stage = _current_plan_stage(stage_steps)
    if current_stage is None:
        return {}
    stage_plan = [_stage_step_context_for_runner(step) for step in stage_steps]
    stage_context: dict[str, object] = {
        **_stage_step_context_for_runner(current_stage),
        "task_id": turn_request.task_id,
        "run_id": turn_request.run_id,
        "turn_index": turn_request.turn_index,
        "stage_plan": stage_plan,
    }
    return {key: value for key, value in stage_context.items() if value not in (None, "", {}, [])}


def _stage_step_context_for_runner(step: AgentStepState) -> dict[str, object]:
    input_payload = dict(step.input_payload or {})
    output_payload = dict(step.output_payload or {})
    stage_context: dict[str, object] = {
        "stage_id": str(input_payload.get("stage_id") or step.capability or ""),
        "title": str(input_payload.get("title") or step.capability or ""),
        "objective": str(input_payload.get("objective") or ""),
        "business_action": str(input_payload.get("business_action") or ""),
        "allowed_capabilities": [str(item) for item in input_payload.get("allowed_capabilities") or []],
        "tool_strategy": dict(input_payload.get("tool_strategy") or {}),
        "ranking_policy": [str(item) for item in input_payload.get("ranking_policy") or []],
        "status": _enum_value(step.status),
        "capability": str(step.capability or ""),
    }
    received_context = input_payload.get("received_context")
    if isinstance(received_context, dict) and received_context:
        stage_context["received_context"] = dict(received_context)
    handoff_payload = output_payload.get("handoff_payload")
    if isinstance(handoff_payload, dict) and handoff_payload:
        stage_context["handoff_payload"] = dict(handoff_payload)
    return {key: value for key, value in stage_context.items() if value not in (None, "", {}, [])}


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
        _ensure_outer_task_stage_plan(turn_request, service=service)
        _mark_outer_task_stage_started(turn_request, service=service)
        try:
            service.get_step(step_id)
        except DurableStateNotFoundError:
            service.add_step(
                task_id=turn_request.task_id,
                step_id=step_id,
                sequence_index=_outer_durable_turn_sequence_index(turn_request),
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

        latest_step_id = _sync_outer_session_workflow_steps(
            turn_request,
            turn_result,
            state=state,
            dependencies=dependencies,
            parent_step_id=step_id,
        )
        _sync_outer_task_stage_execution(
            turn_request,
            turn_result,
            state=state,
            dependencies=dependencies,
            service=service,
        )
        task = service.get_task(turn_result.task_id)
        workflow_run_ids = _append_unique_strings(
            (task.output_payload or {}).get("workflow_run_ids"),
            state.workflow_run_id,
        )
        task.status = task_status
        task.current_step_id = latest_step_id or step_id
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


def _outer_durable_stage_step_id(task_id: str, stage_index: int) -> str:
    candidate = f"{task_id}:stage-{stage_index}"
    if len(candidate) <= 64:
        return candidate
    return f"{task_id[:56]}:s{stage_index}"[:64]


def _ensure_outer_task_stage_plan(turn_request: OuterSessionRunRequest, *, service: DurableStateService) -> None:
    for index, stage in enumerate(DEFAULT_TASK_PLAN_STAGES, start=1):
        step_id = _outer_durable_stage_step_id(turn_request.task_id, index)
        try:
            service.get_step(step_id)
            continue
        except DurableStateNotFoundError:
            pass
        service.add_step(
            task_id=turn_request.task_id,
            step_id=step_id,
            sequence_index=index * 100,
            step_type="workflow_plan_stage",
            status=AgentStepStatus.PENDING,
            executor_type="planner",
            executor_name="offermaster_stage_planner",
            capability=str(stage["capability"]),
            input_payload={
                "stage_id": stage["stage_id"],
                "stage_index": index,
                "title": stage["title"],
                "objective": stage["objective"],
                "business_action": stage.get("business_action"),
                "allowed_capabilities": list(stage.get("allowed_capabilities") or []),
                "tool_strategy": dict(stage.get("tool_strategy") or {}),
                "ranking_policy": list(stage.get("ranking_policy") or []),
                "depends_on": list(stage.get("depends_on") or []),
                "user_goal": turn_request.user_goal,
            },
            output_payload={"planner": "default_mvp", "status": AgentStepStatus.PENDING.value},
        )
    task = service.get_task(turn_request.task_id)
    task.input_payload = {
        **dict(task.input_payload or {}),
        "plan_stage_count": len(DEFAULT_TASK_PLAN_STAGES),
        "plan_planner": "default_mvp",
    }
    service.repository.update_task(task)


def _mark_outer_task_stage_started(turn_request: OuterSessionRunRequest, *, service: DurableStateService) -> None:
    step = _get_outer_task_stage_step(turn_request.task_id, 1, service=service)
    if step is None:
        return
    if _enum_value(step.status) in {AgentStepStatus.SUCCEEDED.value, AgentStepStatus.SKIPPED.value}:
        return
    _set_plan_stage_status(
        step,
        AgentStepStatus.RUNNING,
        service=service,
        output_payload={
            "execution_status": "running",
            "workflow_run_id": turn_request.run_id,
            "user_message": turn_request.user_message,
        },
    )


def _sync_outer_task_stage_execution(
    turn_request: OuterSessionRunRequest,
    turn_result: OuterSessionTurnResult,
    *,
    state,
    dependencies: AgentGraphDependencies,
    service: DurableStateService,
) -> None:
    if turn_result.status == OuterSessionStatus.WAITING_USER:
        _complete_plan_stage(
            turn_request.task_id,
            1,
            AgentStepStatus.WAITING_USER,
            service=service,
            output_payload={
                "execution_status": "waiting_user",
                "workflow_run_id": state.workflow_run_id,
                "waiting_message": turn_result.waiting_message,
            },
        )
        return

    _complete_plan_stage(
        turn_request.task_id,
        1,
        AgentStepStatus.SUCCEEDED,
        service=service,
        output_payload={"execution_status": "finished", "workflow_run_id": state.workflow_run_id},
    )
    touched_stage_indexes = {1}
    for tool_log in _workflow_tool_logs(state.workflow_run_id, dependencies=dependencies):
        stage_index = _plan_stage_index_for_tool_name(tool_log.tool_name)
        if stage_index is None:
            continue
        touched_stage_indexes.add(stage_index)
        _complete_plan_stage(
            turn_request.task_id,
            stage_index,
            _plan_stage_status_from_tool_status(tool_log.status),
            service=service,
            output_payload={
                "execution_status": _enum_value(tool_log.status),
                "workflow_run_id": state.workflow_run_id,
                "tool_call_log_id": tool_log.id,
                "tool_name": tool_log.tool_name,
                "tool_group": tool_log.tool_group,
                "error": tool_log.error,
            },
            tool_logs=[tool_log],
        )

    if turn_result.status == OuterSessionStatus.FINISHED:
        for stage_index in _plan_stage_indexes_from_loop_history(state):
            if stage_index in touched_stage_indexes or stage_index == 5:
                continue
            touched_stage_indexes.add(stage_index)
            _complete_plan_stage(
                turn_request.task_id,
                stage_index,
                AgentStepStatus.SUCCEEDED,
                service=service,
                output_payload={
                    "execution_status": "model_stage_completed",
                    "workflow_run_id": state.workflow_run_id,
                    "response_mode": state.response_mode,
                },
            )

    final_status = AgentStepStatus.SUCCEEDED if turn_result.status == OuterSessionStatus.FINISHED else AgentStepStatus.FAILED
    _complete_plan_stage(
        turn_request.task_id,
        5,
        final_status,
        service=service,
        output_payload={
            "execution_status": turn_result.status.value,
            "workflow_run_id": state.workflow_run_id,
            "response_mode": state.response_mode,
            "final_answer_preview": _text_preview(turn_result.final_answer or state.final_response),
        },
    )
    touched_stage_indexes.add(5)
    for stage_index in range(1, len(DEFAULT_TASK_PLAN_STAGES) + 1):
        if stage_index in touched_stage_indexes:
            continue
        _skip_pending_plan_stage(
            turn_request.task_id,
            stage_index,
            service=service,
            output_payload={"execution_status": "skipped_not_needed_in_mvp", "workflow_run_id": state.workflow_run_id},
        )
    _sync_plan_stage_handoffs(turn_request.task_id, service=service)


def _get_outer_task_stage_step(task_id: str, stage_index: int, *, service: DurableStateService) -> AgentStepState | None:
    try:
        return service.get_step(_outer_durable_stage_step_id(task_id, stage_index))
    except DurableStateNotFoundError:
        return None


def _complete_plan_stage(
    task_id: str,
    stage_index: int,
    status: AgentStepStatus,
    *,
    service: DurableStateService,
    output_payload: dict[str, object],
    tool_logs: list[ToolCallLog] | None = None,
) -> None:
    step = _get_outer_task_stage_step(task_id, stage_index, service=service)
    if step is None:
        return
    output_payload = _stage_output_with_handoff(step, status, output_payload=output_payload, tool_logs=tool_logs or [])
    _set_plan_stage_status(step, status, service=service, output_payload=output_payload)


def _stage_output_with_handoff(
    step: AgentStepState,
    status: AgentStepStatus,
    *,
    output_payload: dict[str, object],
    tool_logs: list[ToolCallLog],
) -> dict[str, object]:
    if status not in {AgentStepStatus.SUCCEEDED, AgentStepStatus.FAILED, AgentStepStatus.WAITING_USER}:
        return output_payload
    input_payload = dict(step.input_payload or {})
    stage_id = str(input_payload.get("stage_id") or step.capability)
    title = str(input_payload.get("title") or step.capability)
    summary = _stage_handoff_summary(output_payload, tool_logs=tool_logs)
    if not summary and status == AgentStepStatus.WAITING_USER:
        summary = str(output_payload.get("waiting_message") or "")
    if not summary:
        return output_payload
    return {
        **output_payload,
        "handoff_payload": {
            "source_stage_id": stage_id,
            "source_stage_title": title,
            "source_step_id": step.id,
            "status": status.value,
            "workflow_run_id": output_payload.get("workflow_run_id"),
            "tool_call_log_ids": [str(log.id) for log in tool_logs],
            "tool_names": [str(log.tool_name) for log in tool_logs],
            "summary": summary,
        },
    }


def _sync_plan_stage_handoffs(task_id: str, *, service: DurableStateService) -> None:
    stage_steps = [
        step
        for index in range(1, len(DEFAULT_TASK_PLAN_STAGES) + 1)
        if (step := _get_outer_task_stage_step(task_id, index, service=service)) is not None
    ]
    handoff_by_stage_id: dict[str, dict[str, object]] = {}
    for step in stage_steps:
        input_payload = dict(step.input_payload or {})
        output_payload = dict(step.output_payload or {})
        stage_id = str(input_payload.get("stage_id") or step.capability)
        handoff = output_payload.get("handoff_payload")
        if isinstance(handoff, dict):
            handoff_by_stage_id[stage_id] = dict(handoff)

    for step in stage_steps:
        input_payload = dict(step.input_payload or {})
        depends_on = [str(item) for item in input_payload.get("depends_on") or []]
        upstream_handoffs = [handoff_by_stage_id[stage_id] for stage_id in depends_on if stage_id in handoff_by_stage_id]
        if not upstream_handoffs:
            continue
        received_context = _received_context_from_handoffs(upstream_handoffs)
        if input_payload.get("received_context") == received_context:
            continue
        step.input_payload = {**input_payload, "received_context": received_context}
        service.repository.update_step(step)


def _received_context_from_handoffs(handoffs: list[dict[str, object]]) -> dict[str, object]:
    summaries = [str(handoff.get("summary") or "").strip() for handoff in handoffs]
    summaries = [summary for summary in summaries if summary]
    tool_names: list[str] = []
    tool_call_log_ids: list[str] = []
    from_step_ids: list[str] = []
    upstream_stage_ids: list[str] = []
    for handoff in handoffs:
        stage_id = str(handoff.get("source_stage_id") or "").strip()
        step_id = str(handoff.get("source_step_id") or "").strip()
        if stage_id:
            upstream_stage_ids = _append_unique_strings(upstream_stage_ids, stage_id)
        if step_id:
            from_step_ids = _append_unique_strings(from_step_ids, step_id)
        for tool_name in handoff.get("tool_names") or []:
            tool_names = _append_unique_strings(tool_names, tool_name)
        for log_id in handoff.get("tool_call_log_ids") or []:
            tool_call_log_ids = _append_unique_strings(tool_call_log_ids, log_id)
    return {
        "upstream_stage_ids": upstream_stage_ids,
        "from_step_ids": from_step_ids,
        "tool_call_log_ids": tool_call_log_ids,
        "tool_names": tool_names,
        "summary": "\n".join(summaries),
    }


def _stage_handoff_summary(output_payload: dict[str, object], *, tool_logs: list[ToolCallLog]) -> str:
    final_preview = str(output_payload.get("final_answer_preview") or "").strip()
    if final_preview:
        return final_preview
    waiting_message = str(output_payload.get("waiting_message") or "").strip()
    if waiting_message:
        return waiting_message
    summaries = [_tool_log_handoff_summary(log) for log in tool_logs]
    summaries = [summary for summary in summaries if summary]
    return "\n".join(summaries)


def _tool_log_handoff_summary(tool_log: ToolCallLog) -> str:
    if tool_log.error:
        return f"工具 {tool_log.tool_name} 执行失败：{tool_log.error}"
    output_payload = tool_log.output_payload if isinstance(tool_log.output_payload, dict) else {}
    result_payload = output_payload.get("result") if isinstance(output_payload.get("result"), dict) else {}
    local_result = result_payload.get("result") if isinstance(result_payload.get("result"), dict) else {}
    envelope = result_payload.get("result_envelope") if isinstance(result_payload.get("result_envelope"), dict) else {}
    if tool_log.tool_name == "local.company_database_overview" and local_result:
        return _local_company_overview_handoff_summary(local_result)
    if tool_log.tool_name == "local.job_source_overview" and local_result:
        return _local_job_source_handoff_summary(local_result)
    envelope_summary = str(envelope.get("summary") or "").strip()
    if envelope_summary:
        return _text_preview(envelope_summary, limit=360)
    if result_payload:
        return _text_preview(json.dumps(result_payload, ensure_ascii=False, default=str), limit=360)
    return ""


def _local_company_overview_handoff_summary(result: dict[str, object]) -> str:
    company_count = _int_payload_value(result.get("company_count"))
    job_count = _int_payload_value(result.get("job_count"))
    lead_count = _int_payload_value(result.get("job_lead_count"))
    lead_company_count = _int_payload_value(result.get("job_lead_company_count"))
    signal_count = _int_payload_value(result.get("recruiting_signal_count"))
    signal_company_count = _int_payload_value(result.get("recruiting_signal_company_count"))
    return (
        f"本地候选信息：正式企业表 {company_count} 家，正式岗位 {job_count} 条；"
        f"岗位线索 {lead_count} 条，去重企业 {lead_company_count} 家；"
        f"公司校招来源 {signal_count} 条，去重企业 {signal_company_count} 家。"
    )


def _local_job_source_handoff_summary(result: dict[str, object]) -> str:
    source_count = _int_payload_value(result.get("source_count"))
    enabled_count = _int_payload_value(result.get("enabled_source_count"))
    unsynced_count = _int_payload_value(result.get("unsynced_source_count"))
    return f"岗位来源信息：来源 {source_count} 个，已启用 {enabled_count} 个，待同步 {unsynced_count} 个。"


def _int_payload_value(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _skip_pending_plan_stage(
    task_id: str,
    stage_index: int,
    *,
    service: DurableStateService,
    output_payload: dict[str, object],
) -> None:
    step = _get_outer_task_stage_step(task_id, stage_index, service=service)
    if step is None:
        return
    if _enum_value(step.status) != AgentStepStatus.PENDING.value:
        return
    _set_plan_stage_status(step, AgentStepStatus.SKIPPED, service=service, output_payload=output_payload)


def _set_plan_stage_status(
    step: AgentStepState,
    status: AgentStepStatus,
    *,
    service: DurableStateService,
    output_payload: dict[str, object],
) -> None:
    now = utc_now()
    step.status = status
    if step.started_at is None and status in {
        AgentStepStatus.RUNNING,
        AgentStepStatus.WAITING_USER,
        AgentStepStatus.SUCCEEDED,
        AgentStepStatus.FAILED,
        AgentStepStatus.SKIPPED,
        AgentStepStatus.BLOCKED,
    }:
        step.started_at = now
    if status in {AgentStepStatus.SUCCEEDED, AgentStepStatus.FAILED, AgentStepStatus.SKIPPED, AgentStepStatus.BLOCKED}:
        step.completed_at = now
    if status == AgentStepStatus.RUNNING:
        step.completed_at = None
    step.output_payload = {
        **dict(step.output_payload or {}),
        **dict(output_payload or {}),
        "status": status.value,
    }
    service.repository.update_step(step)


def _plan_stage_index_for_tool_name(tool_name: str | None) -> int | None:
    name = str(tool_name or "")
    if name.startswith("local.") or name.startswith("offerio."):
        return 2
    if name.startswith("external.") or "web_search" in name or "xiaohongshu" in name or "wechat" in name:
        return 3
    return None


def _plan_stage_indexes_from_loop_history(state) -> list[int]:
    metadata = state.context_metadata if isinstance(state.context_metadata, dict) else {}
    loop_metadata = metadata.get("tool_choice_loop") if isinstance(metadata.get("tool_choice_loop"), dict) else {}
    nested_metadata = loop_metadata.get("metadata") if isinstance(loop_metadata.get("metadata"), dict) else {}
    history = nested_metadata.get("stage_context_history")
    if not isinstance(history, list):
        return []
    indexes: list[int] = []
    for stage_id in history:
        stage_index = _plan_stage_index_for_stage_id(str(stage_id))
        if stage_index is not None and stage_index not in indexes:
            indexes.append(stage_index)
    return indexes


def _plan_stage_index_for_stage_id(stage_id: str) -> int | None:
    for index, stage in enumerate(DEFAULT_TASK_PLAN_STAGES, start=1):
        if str(stage.get("stage_id") or "") == stage_id:
            return index
    return None


def _plan_stage_status_from_tool_status(status_value: object) -> AgentStepStatus:
    status = _enum_value(status_value)
    if status == "succeeded":
        return AgentStepStatus.SUCCEEDED
    if status in {"failed", "blocked"}:
        return AgentStepStatus.FAILED
    return AgentStepStatus.RUNNING


def _outer_durable_turn_sequence_index(turn_request: OuterSessionRunRequest) -> int:
    return max(1, turn_request.turn_index) * 1000


def _outer_durable_child_step_id(turn_request: OuterSessionRunRequest, suffix: str) -> str:
    candidate = f"{_outer_durable_turn_step_id(turn_request)}:{suffix}"
    if len(candidate) <= 64:
        return candidate
    return f"{turn_request.task_id[:48]}:{suffix}"[:64]


def _sync_outer_session_workflow_steps(
    turn_request: OuterSessionRunRequest,
    turn_result: OuterSessionTurnResult,
    *,
    state,
    dependencies: AgentGraphDependencies,
    parent_step_id: str,
) -> str | None:
    service = dependencies.durable_state_service
    if service is None:
        return None
    latest_step_id: str | None = None
    base_sequence = _outer_durable_turn_sequence_index(turn_request)
    latest_step_id = _ensure_outer_workflow_context_step(
        turn_request,
        state=state,
        service=service,
        parent_step_id=parent_step_id,
        sequence_index=base_sequence + 1,
    ) or latest_step_id
    for index, tool_log in enumerate(_workflow_tool_logs(state.workflow_run_id, dependencies=dependencies), start=1):
        latest_step_id = _ensure_outer_workflow_tool_step(
            turn_request,
            tool_log,
            service=service,
            parent_step_id=parent_step_id,
            sequence_index=base_sequence + 100 + index,
            tool_index=index,
        ) or latest_step_id
    latest_step_id = _ensure_outer_workflow_final_step(
        turn_request,
        turn_result,
        state=state,
        service=service,
        parent_step_id=parent_step_id,
        sequence_index=base_sequence + 900,
    ) or latest_step_id
    return latest_step_id


def _ensure_outer_workflow_context_step(
    turn_request: OuterSessionRunRequest,
    *,
    state,
    service: DurableStateService,
    parent_step_id: str,
    sequence_index: int,
) -> str | None:
    step_id = _outer_durable_child_step_id(turn_request, "ctx")
    try:
        service.get_step(step_id)
        return step_id
    except DurableStateNotFoundError:
        pass
    service.add_step(
        task_id=turn_request.task_id,
        step_id=step_id,
        parent_step_id=parent_step_id,
        sequence_index=sequence_index,
        step_type="workflow_context",
        executor_type="runtime",
        executor_name="offermaster_context_builder",
        capability="agent.context_builder",
        input_payload={
            "workflow_run_id": state.workflow_run_id,
            "agent_run_id": state.agent_run_id,
            "user_message": state.user_message,
        },
    )
    service.mark_step_succeeded(
        step_id,
        output_payload={
            "latest_summary_id": state.latest_summary_id,
            "loaded_session_history_count": len(state.loaded_session_history_ids),
            "loaded_memory_count": len(state.loaded_memory_ids),
            "loaded_skill_count": len(state.loaded_skill_ids),
            "token_estimate": state.token_estimate,
            "need_compaction": state.need_compaction,
        },
    )
    return step_id


def _ensure_outer_workflow_tool_step(
    turn_request: OuterSessionRunRequest,
    tool_log: ToolCallLog,
    *,
    service: DurableStateService,
    parent_step_id: str,
    sequence_index: int,
    tool_index: int,
) -> str | None:
    step_id = _outer_durable_child_step_id(turn_request, f"tool-{tool_index}")
    try:
        service.get_step(step_id)
        return step_id
    except DurableStateNotFoundError:
        pass
    status = _enum_value(tool_log.status)
    service.add_step(
        task_id=turn_request.task_id,
        step_id=step_id,
        parent_step_id=parent_step_id,
        sequence_index=sequence_index,
        step_type="workflow_tool_call",
        executor_type="tool_registry",
        executor_name=str(tool_log.tool_group or "agent_tool_registry"),
        capability=tool_log.tool_name,
        input_payload={
            "workflow_run_id": tool_log.workflow_run_id,
            "tool_name": tool_log.tool_name,
            "tool_group": tool_log.tool_group,
            "tool_input": dict(tool_log.input_payload or {}),
        },
        tool_call_log_id=tool_log.id,
    )
    output_payload = {
        "tool_call_log_id": tool_log.id,
        "tool_name": tool_log.tool_name,
        "tool_group": tool_log.tool_group,
        "tool_status": status,
        "error": tool_log.error,
        "duration_ms": tool_log.duration_ms,
        "tool_output": dict(tool_log.output_payload or {}),
    }
    if status == "succeeded":
        service.mark_step_succeeded(step_id, tool_call_log_id=tool_log.id, output_payload=output_payload)
    elif status in {"blocked", "failed"}:
        service.mark_step_failed(step_id, output_payload=output_payload)
    else:
        service.mark_step_running(step_id)
    return step_id


def _ensure_outer_workflow_final_step(
    turn_request: OuterSessionRunRequest,
    turn_result: OuterSessionTurnResult,
    *,
    state,
    service: DurableStateService,
    parent_step_id: str,
    sequence_index: int,
) -> str | None:
    step_id = _outer_durable_child_step_id(turn_request, "final")
    try:
        service.get_step(step_id)
        return step_id
    except DurableStateNotFoundError:
        pass
    is_waiting = turn_result.status == OuterSessionStatus.WAITING_USER
    is_failed = turn_result.status == OuterSessionStatus.FAILED
    step_type = "workflow_waiting_user" if is_waiting else "workflow_final_response"
    service.add_step(
        task_id=turn_request.task_id,
        step_id=step_id,
        parent_step_id=parent_step_id,
        sequence_index=sequence_index,
        step_type=step_type,
        executor_type="runtime",
        executor_name="offermaster_finalizer",
        capability="agent.final_response",
        input_payload={
            "workflow_run_id": state.workflow_run_id,
            "current_step": state.current_step,
            "requires_user_action": turn_result.requires_user_action,
        },
        approval_request_id=state.approval_request_id,
    )
    output_payload = {
        "workflow_run_id": state.workflow_run_id,
        "response_mode": state.response_mode,
        "current_step": state.current_step,
        "requires_user_action": turn_result.requires_user_action,
        "waiting_message": turn_result.waiting_message,
        "final_answer_preview": _text_preview(turn_result.final_answer or state.final_response),
    }
    if is_waiting:
        service.mark_step_waiting_user(step_id, approval_request_id=state.approval_request_id, output_payload=output_payload)
    elif is_failed:
        service.mark_step_failed(step_id, output_payload=output_payload)
    else:
        service.mark_step_succeeded(step_id, output_payload=output_payload)
    return step_id


def _workflow_tool_logs(workflow_run_id: str, *, dependencies: AgentGraphDependencies) -> list[ToolCallLog]:
    if dependencies.db_session is None:
        return []
    return list(
        dependencies.db_session.scalars(
            select(ToolCallLog)
            .where(ToolCallLog.workflow_run_id == workflow_run_id)
            .order_by(ToolCallLog.created_at.asc(), ToolCallLog.id.asc())
        ).all()
    )


def _text_preview(value: object, limit: int = 500) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


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
        input_preview = _runtime_input_preview(metadata)
        result_summary = _runtime_result_summary(metadata)
        evidence = _runtime_evidence_from_result_summary(result_summary)
        decision_reason = str(raw_entry.get("decision_reason") or "").strip()
        if decision_reason:
            events.append(
                _tool_event_payload(
                    "reasoning_summary",
                    state=state,
                    step_index=step_index,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    status="thinking",
                    summary=decision_reason,
                    tool_input_keys=tool_input_keys,
                )
            )
        if input_preview:
            events.append(
                _tool_event_payload(
                    "tool_input_preview",
                    state=state,
                    step_index=step_index,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    status="running",
                    summary=_runtime_input_preview_summary(input_preview),
                    tool_input_keys=tool_input_keys,
                    input_preview=input_preview,
                )
            )
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
        if result_summary:
            events.append(
                _tool_event_payload(
                    "tool_result_summary",
                    state=state,
                    step_index=step_index,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    status=str(raw_entry.get("observation_status") or "unknown"),
                    summary=_runtime_result_summary_text(result_summary),
                    tool_input_keys=tool_input_keys,
                    result_summary=result_summary,
                )
            )
        reflection = metadata.get("reflection") if isinstance(metadata.get("reflection"), dict) else None
        if isinstance(reflection, dict):
            events.append(
                _tool_event_payload(
                    "reflection_evaluation",
                    state=state,
                    step_index=step_index,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    status=str(reflection.get("next_action") or "observed"),
                    summary=str(reflection.get("reason") or "主 agent 已评估这次工具结果。"),
                    tool_input_keys=tool_input_keys,
                    reflection=reflection,
                )
            )
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
        if evidence:
            events.append(
                _tool_event_payload(
                    "evidence_selected",
                    state=state,
                    step_index=step_index,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    status="succeeded",
                    summary=_runtime_evidence_summary(evidence),
                    tool_input_keys=tool_input_keys,
                    evidence=evidence,
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
    input_preview: dict[str, object] | None = None,
    result_summary: dict[str, object] | None = None,
    evidence: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    labels = {
        "reasoning_summary": "思考摘要",
        "tool_input_preview": "工具输入",
        "tool_started": "工具开始",
        "tool_finished": "工具完成",
        "tool_result_summary": "结果摘要",
        "reflection_evaluation": "反思判断",
        "tool_reflection_retry": "准备重试",
        "evidence_selected": "证据选择",
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
    if input_preview is not None:
        payload["input_preview"] = dict(input_preview)
    if result_summary is not None:
        payload["result_summary"] = dict(result_summary)
    if evidence is not None:
        payload["evidence"] = [dict(item) for item in evidence]
    return payload


def _runtime_input_preview(metadata: dict[str, object]) -> dict[str, object] | None:
    tool_input = metadata.get("tool_input") if isinstance(metadata.get("tool_input"), dict) else None
    if tool_input is None:
        observation = metadata.get("observation") if isinstance(metadata.get("observation"), dict) else None
        observation_metadata = observation.get("metadata") if isinstance(observation, dict) and isinstance(observation.get("metadata"), dict) else None
        tool_input = observation_metadata.get("tool_input") if isinstance(observation_metadata, dict) and isinstance(observation_metadata.get("tool_input"), dict) else None
    if not isinstance(tool_input, dict):
        return None
    preview: dict[str, object] = {}
    for key, value in tool_input.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            preview[str(key)] = value
    return preview or None


def _runtime_result_summary(metadata: dict[str, object]) -> dict[str, object] | None:
    summary = metadata.get("result_observation") if isinstance(metadata.get("result_observation"), dict) else None
    if summary is None:
        summary = metadata.get("result_summary") if isinstance(metadata.get("result_summary"), dict) else None
    if not isinstance(summary, dict):
        return None
    clean: dict[str, object] = {}
    for key in ("result_count", "source_count"):
        value = summary.get(key)
        if isinstance(value, int):
            clean[key] = value
    domains = summary.get("source_domains")
    if isinstance(domains, list):
        clean["source_domains"] = [str(domain) for domain in domains if str(domain).strip()][:8]
    evidence = summary.get("evidence")
    if isinstance(evidence, list):
        clean["evidence"] = [item for item in (_clean_evidence_item(raw) for raw in evidence) if item]
    return clean or None


def _runtime_evidence_from_result_summary(result_summary: dict[str, object] | None) -> list[dict[str, object]]:
    evidence = result_summary.get("evidence") if isinstance(result_summary, dict) else None
    if not isinstance(evidence, list):
        return []
    return [dict(item) for item in evidence if isinstance(item, dict)]


def _clean_evidence_item(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        return {}
    title = str(raw.get("title") or raw.get("name") or "证据来源").strip()
    url = str(raw.get("url") or raw.get("href") or "").strip()
    if not title and not url:
        return {}
    item: dict[str, object] = {"title": title or "证据来源"}
    if url:
        item["url"] = url
    return item


def _runtime_input_preview_summary(input_preview: dict[str, object]) -> str:
    query = str(input_preview.get("query") or "").strip()
    if query:
        return f"准备使用关键词：{query}。"
    keys = "、".join(input_preview.keys())
    return f"准备使用工具输入字段：{keys}。" if keys else "准备调用工具。"


def _runtime_result_summary_text(result_summary: dict[str, object]) -> str:
    result_count = result_summary.get("result_count")
    source_count = result_summary.get("source_count")
    domains = result_summary.get("source_domains") if isinstance(result_summary.get("source_domains"), list) else []
    parts: list[str] = []
    if isinstance(result_count, int):
        parts.append(f"找到 {result_count} 条结果")
    if isinstance(source_count, int):
        parts.append(f"覆盖 {source_count} 个来源")
    if domains:
        parts.append("来源包括 " + "、".join(str(domain) for domain in domains[:3]))
    return "，".join(parts) + "。" if parts else "工具结果已整理为可观察摘要。"


def _runtime_evidence_summary(evidence: list[dict[str, object]]) -> str:
    titles = [str(item.get("title") or "证据来源").strip() for item in evidence[:3]]
    titles = [title for title in titles if title]
    return f"已选择 {len(evidence)} 条证据：" + "、".join(titles) + "。" if titles else f"已选择 {len(evidence)} 条证据。"


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
    if state.current_step in {"wait_confirmation", "wait_user_input"}:
        return True
    if state.response_mode in {
        "clarification_ask_user",
        "capability_route_ask_user",
        "execution_planner_ask_user",
        "tool_input_ask_user",
    }:
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
    memory_repository = AgentMemoryRepository(session)
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
        memory_repository=memory_repository,
        skill_repository=AgentSkillRepository(memory_repository),
        db_session=session,
        llm_client=_build_agent_llm_client(settings),
        intent_detector=_build_agent_intent_detector(settings),
        execution_planner=_build_agent_execution_planner(settings),
        capability_routing_middleware=CapabilityRoutingMiddleware(),
        durable_state_service=DurableStateService(SqlAlchemyDurableStateRepository(session)),
        agent_executors=agent_executors,
        capability_executor_ids=capability_executor_ids,
    )


def _resume_agent_task_response(task_id: str, *, session: Session) -> AgentTaskResumeResponse:
    service = DurableStateService(SqlAlchemyDurableStateRepository(session))
    try:
        result = service.resume_task(task_id)
        response = _agent_task_resume_response_from_result(result, session=session)
    except DurableStateNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    return response


def _recover_and_run_agent_task_response(
    task_id: str,
    *,
    session: Session,
    conversation_service: ConversationService,
) -> AgentTaskRecoverRunResponse:
    durable_service = DurableStateService(SqlAlchemyDurableStateRepository(session))
    try:
        result = durable_service.resume_task(task_id)
        resume_response = _agent_task_resume_response_from_result(result, session=session)
    except DurableStateNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not _recovery_action_is_executable(result):
        session.commit()
        task_read = _agent_task_read_from_session(session, result.task_id)
        return AgentTaskRecoverRunResponse(
            resume=resume_response,
            executed=False,
            task=task_read,
        )

    workflow_result = _run_recovered_agent_task(
        result,
        session=session,
        conversation_service=conversation_service,
    )
    _mark_recovery_resume_step_finished(result, state=workflow_result.state, session=session)
    assistant_message = _append_assistant_message_from_state(conversation_service, workflow_result.state)
    session.flush()
    task_read = _agent_task_read_from_session(session, result.task_id)
    session.commit()
    return AgentTaskRecoverRunResponse(
        resume=_agent_task_resume_response_from_result(result, session=session),
        executed=True,
        assistant_message=AgentMessageRead.model_validate(assistant_message),
        task=task_read,
        context_metadata=_context_metadata_from_state(workflow_result.state),
    )


def _agent_task_resume_response_from_result(result, *, session: Session) -> AgentTaskResumeResponse:
    stage_info = _resume_stage_info_from_result(result, session=session)
    return AgentTaskResumeResponse(
        action=result.action.value,
        task_id=result.task_id,
        source_step_id=result.source_step_id,
        resume_step_id=result.resume_step_id,
        resume_stage_id=stage_info.get("resume_stage_id"),
        resume_stage_title=stage_info.get("resume_stage_title"),
        resume_stage_step_id=stage_info.get("resume_stage_step_id"),
        resume_stage_status=stage_info.get("resume_stage_status"),
        resume_stage_capability=stage_info.get("resume_stage_capability"),
        reason=result.reason,
        approval_request_id=result.approval_request_id,
        executor_type=result.executor_type,
        executor_name=result.executor_name,
        capability=result.capability,
        payload=result.payload,
        requires_user_action=result.requires_user_action,
        task=_agent_task_read_from_session(session, result.task_id),
    )


def _resume_stage_info_from_result(result, *, session: Session) -> dict[str, str | None]:
    source_step = session.get(AgentStepState, result.source_step_id) if result.source_step_id else None
    stage_steps = _agent_task_plan_stage_steps(session, result.task_id)
    problem_stage = _first_problem_plan_stage(stage_steps)
    if problem_stage is not None:
        return _resume_stage_info_from_stage_step(problem_stage)

    stage_index = _plan_stage_index_for_resume_source(source_step, result.capability)
    if stage_index is None:
        return {}
    for stage_step in stage_steps:
        if _stage_index_from_step(stage_step) == stage_index:
            return _resume_stage_info_from_stage_step(stage_step)
    return _resume_stage_info_from_default_stage(stage_index)


def _first_problem_plan_stage(stage_steps: list[AgentStepState]) -> AgentStepState | None:
    for status in (AgentStepStatus.FAILED.value, AgentStepStatus.WAITING_USER.value, AgentStepStatus.RUNNING.value):
        for step in stage_steps:
            if _enum_value(step.status) == status:
                return step
    return None


def _resume_stage_info_from_stage_step(step: AgentStepState) -> dict[str, str | None]:
    payload = dict(step.input_payload or {})
    return {
        "resume_stage_id": str(payload.get("stage_id") or step.capability),
        "resume_stage_title": str(payload.get("title") or step.capability),
        "resume_stage_step_id": step.id,
        "resume_stage_status": _enum_value(step.status),
        "resume_stage_capability": step.capability,
    }


def _resume_stage_info_from_default_stage(stage_index: int) -> dict[str, str | None]:
    if stage_index < 1 or stage_index > len(DEFAULT_TASK_PLAN_STAGES):
        return {}
    stage = DEFAULT_TASK_PLAN_STAGES[stage_index - 1]
    return {
        "resume_stage_id": str(stage["stage_id"]),
        "resume_stage_title": str(stage["title"]),
        "resume_stage_step_id": None,
        "resume_stage_status": None,
        "resume_stage_capability": str(stage["capability"]),
    }


def _plan_stage_index_for_resume_source(source_step: AgentStepState | None, fallback_capability: str | None) -> int | None:
    if source_step is not None and source_step.step_type == "workflow_plan_stage":
        return _stage_index_from_step(source_step)
    capability = str((source_step.capability if source_step is not None else fallback_capability) or "")
    step_type = str(source_step.step_type if source_step is not None else "")
    output_payload = dict(source_step.output_payload or {}) if source_step is not None else {}
    if capability in {"agent.outer_session", "agent.context_builder"} or step_type == "workflow_context":
        return 1
    if step_type in {"workflow_waiting_user"}:
        return 1
    if capability == "agent.final_response" or step_type == "workflow_final_response":
        if output_payload.get("requires_user_action") or str(output_payload.get("response_mode") or "").endswith("ask_user"):
            return 1
        return 5
    return _plan_stage_index_for_tool_name(capability)


def _stage_index_from_step(step: AgentStepState) -> int | None:
    payload = dict(step.input_payload or {})
    try:
        return int(payload.get("stage_index") or 0) or None
    except (TypeError, ValueError):
        return None


def _recovery_action_is_executable(result) -> bool:
    return result.action.value in {"retry_failed_step", "reissue_executor_task"} and bool(result.capability)


def _enqueue_agent_session_task_followup(
    task: AgentTaskState,
    *,
    request: AgentTaskFollowupRequest,
    session: Session,
    conversation_service: ConversationService,
) -> AgentTaskFollowupResponse:
    content_text = str(request.content_text or "").strip()
    if not content_text:
        raise HTTPException(status_code=400, detail="Follow-up content cannot be empty")
    if _enum_value(task.status) in {AgentTaskStatus.SUCCEEDED.value, AgentTaskStatus.CANCELED.value}:
        raise HTTPException(status_code=409, detail="Cannot add follow-up to a completed agent task")

    store = _AgentSessionOuterLoopStore(conversation_service, task.conversation_session_id)
    outer_state = store.get(task.conversation_session_id)
    if outer_state is None or outer_state.active_task_id != task.id:
        outer_state = _outer_session_state_from_task(task)

    outer_state.user_followups.append(content_text)
    outer_state.metadata = {
        **dict(outer_state.metadata),
        "pending_followup_count": len(outer_state.user_followups),
        "latest_followup_text": content_text,
        "latest_followup_at": utc_now().isoformat(),
    }
    store.save(outer_state)

    task.output_payload = {
        **dict(task.output_payload or {}),
        "pending_followups": list(outer_state.user_followups),
        "pending_followup_count": len(outer_state.user_followups),
    }
    session.flush()
    response = AgentTaskFollowupResponse(
        task_id=task.id,
        queued_count=len(outer_state.user_followups),
        user_followups=list(outer_state.user_followups),
        task=_agent_task_read_from_session(session, task.id),
    )
    session.commit()
    return response


def _outer_session_state_from_task(task: AgentTaskState) -> OuterSessionState:
    return OuterSessionState(
        session_id=task.conversation_session_id,
        active_task_id=task.id,
        user_goal=str(task.user_goal or "继续执行任务").strip() or "继续执行任务",
        status=_outer_status_from_agent_task_status(task.status),
        run_count=0,
        active_run_id=str(task.root_workflow_run_id or "") or None,
        metadata={
            "durable_task_id": task.id,
            "durable_status": _enum_value(task.status),
        },
    )


def _outer_status_from_agent_task_status(status_value: object) -> OuterSessionStatus:
    status_text = _enum_value(status_value)
    if status_text == AgentTaskStatus.WAITING_USER.value:
        return OuterSessionStatus.WAITING_USER
    if status_text == AgentTaskStatus.FAILED.value:
        return OuterSessionStatus.FAILED
    if status_text in {AgentTaskStatus.SUCCEEDED.value, AgentTaskStatus.CANCELED.value}:
        return OuterSessionStatus.FINISHED
    return OuterSessionStatus.RUNNING


def _consume_outer_session_followups(
    *,
    conversation_service: ConversationService,
    session_id: str,
    task_id: str,
) -> list[str]:
    store = _AgentSessionOuterLoopStore(conversation_service, session_id)
    outer_state = store.get(session_id)
    if outer_state is None or outer_state.active_task_id != task_id:
        return []
    followups = [str(item).strip() for item in outer_state.user_followups if str(item).strip()]
    if not followups:
        return []
    outer_state.user_followups = []
    outer_state.metadata = {
        **dict(outer_state.metadata),
        "pending_followup_count": 0,
        "last_consumed_followups": followups,
        "last_consumed_followup_count": len(followups),
        "last_consumed_followup_at": utc_now().isoformat(),
    }
    store.save(outer_state)
    return followups


def _user_message_with_followups(user_goal: str, followups: list[str]) -> str:
    base = str(user_goal or "继续执行恢复任务").strip() or "继续执行恢复任务"
    if not followups:
        return base
    followup_lines = "\n".join(f"- {item}" for item in followups)
    return f"{base}\n\n运行中用户补充要求：\n{followup_lines}"


def _user_message_with_recovery_stage_context(user_message: str, stage_info: dict[str, str | None]) -> str:
    stage_id = str(stage_info.get("resume_stage_id") or "").strip()
    stage_title = str(stage_info.get("resume_stage_title") or "").strip()
    if not stage_id and not stage_title:
        return user_message
    lines = []
    if stage_title:
        lines.append(f"- 恢复阶段：{stage_title}")
    if stage_id:
        lines.append(f"- 阶段标识：{stage_id}")
    return f"{user_message}\n\n阶段级恢复上下文：\n" + "\n".join(lines)


def _run_recovered_agent_task(
    result,
    *,
    session: Session,
    conversation_service: ConversationService,
) -> AgentWorkflowResult:
    task = session.get(AgentTaskState, result.task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Agent task state not found: {result.task_id}")
    dependencies = _agent_graph_dependencies(session, conversation_service)
    followups = _consume_outer_session_followups(
        conversation_service=conversation_service,
        session_id=task.conversation_session_id,
        task_id=result.task_id,
    )
    if followups:
        task.output_payload = {
            **dict(task.output_payload or {}),
            "pending_followups": [],
            "pending_followup_count": 0,
            "last_consumed_followups": followups,
            "last_consumed_followup_at": utc_now().isoformat(),
        }
        session.flush()
    user_message = _user_message_with_followups(str(task.user_goal or ""), followups)
    user_message = _user_message_with_recovery_stage_context(
        user_message,
        _resume_stage_info_from_result(result, session=session),
    )
    recover_via_stage_loop = _prepare_failed_stage_for_loop_recovery(result.task_id, session=session)
    command = AgentRunCommand(
        session_id=task.conversation_session_id,
        user_message=user_message,
        requested_tool_name=None if recover_via_stage_loop else result.capability,
        source_type="agent_chat",
        user_confirmed=True,
        tool_input={} if recover_via_stage_loop else dict(result.payload or {}),
    )
    return _run_agent_workflow_with_outer_session(
        command,
        dependencies=dependencies,
        conversation_service=conversation_service,
        task_id=result.task_id,
    )


def _prepare_failed_stage_for_loop_recovery(task_id: str, *, session: Session) -> bool:
    stage_steps = _agent_task_plan_stage_steps(session, task_id)
    problem_stage = _first_problem_plan_stage(stage_steps)
    if problem_stage is None or _enum_value(problem_stage.status) != AgentStepStatus.FAILED.value:
        return False
    problem_index = _stage_index_from_step(problem_stage)
    if problem_index is None:
        return False

    problem_stage.status = AgentStepStatus.RUNNING
    problem_stage.completed_at = None
    problem_stage.output_payload = {
        **dict(problem_stage.output_payload or {}),
        "execution_status": "recovering_failed_stage",
        "recovery_started_at": utc_now().isoformat(),
    }
    session.add(problem_stage)

    for stage_step in stage_steps:
        stage_index = _stage_index_from_step(stage_step)
        if stage_index is None or stage_index <= problem_index:
            continue
        if _enum_value(stage_step.status) in {AgentStepStatus.PENDING.value, AgentStepStatus.RUNNING.value}:
            continue
        stage_step.status = AgentStepStatus.PENDING
        stage_step.completed_at = None
        stage_step.output_payload = {
            **dict(stage_step.output_payload or {}),
            "execution_status": "pending_after_stage_recovery",
        }
        session.add(stage_step)

    task = session.get(AgentTaskState, task_id)
    if task is not None:
        task.status = AgentTaskStatus.RUNNING
        task.current_step_id = problem_stage.id
        task.completed_at = None
        task.output_payload = {
            **dict(task.output_payload or {}),
            "recovery_mode": "stage_loop",
            "recovery_stage_id": str((problem_stage.input_payload or {}).get("stage_id") or problem_stage.capability),
        }
        session.add(task)
    session.flush()
    return True


def _mark_recovery_resume_step_finished(result, *, state, session: Session) -> None:
    if not result.resume_step_id:
        return
    service = DurableStateService(SqlAlchemyDurableStateRepository(session))
    try:
        step = service.get_step(result.resume_step_id)
    except DurableStateNotFoundError:
        return
    stage_info = _resume_stage_info_from_result(result, session=session)
    step.status = AgentStepStatus.WAITING_USER if _state_requires_outer_user_action(state) else AgentStepStatus.SUCCEEDED
    step.completed_at = utc_now()
    step.output_payload = {
        **dict(step.output_payload or {}),
        "recovery_workflow_run_id": state.workflow_run_id,
        "response_mode": state.response_mode,
        "current_step": state.current_step,
        "final_answer_preview": _text_preview(state.final_response),
        "resume_stage_id": stage_info.get("resume_stage_id"),
        "resume_stage_title": stage_info.get("resume_stage_title"),
        "resume_stage_step_id": stage_info.get("resume_stage_step_id"),
    }
    service.repository.update_step(step)


def _latest_agent_session_task(session: Session, session_id: str) -> AgentTaskState | None:
    return session.scalars(
        select(AgentTaskState)
        .where(AgentTaskState.conversation_session_id == session_id)
        .where(AgentTaskState.task_type == "outer_session_task")
        .order_by(AgentTaskState.updated_at.desc(), AgentTaskState.created_at.desc(), AgentTaskState.id.desc())
        .limit(1)
    ).first()


def _agent_task_read_from_session(session: Session, task_id: str) -> AgentTaskRead:
    task = session.get(AgentTaskState, task_id)
    if task is None:
        raise DurableStateNotFoundError(f"Agent task state not found: {task_id}")
    return _agent_task_read(task, steps=_agent_task_steps(session, task_id))


def _agent_task_steps(session: Session, task_id: str) -> list[AgentStepState]:
    return list(
        session.scalars(
            select(AgentStepState)
            .where(AgentStepState.task_id == task_id)
            .order_by(AgentStepState.sequence_index.asc(), AgentStepState.created_at.asc(), AgentStepState.id.asc())
        ).all()
    )


def _agent_task_plan_response(task: AgentTaskState, *, session: Session) -> AgentTaskPlanResponse:
    stage_steps = _agent_task_plan_stage_steps(session, task.id)
    current_stage = _current_plan_stage(stage_steps)
    return AgentTaskPlanResponse(
        task_id=task.id,
        user_goal=task.user_goal,
        current_stage_id=current_stage.capability if current_stage is not None else None,
        stages=[_agent_task_plan_stage_read(step) for step in stage_steps],
    )


def _agent_task_plan_stage_steps(session: Session, task_id: str) -> list[AgentStepState]:
    return list(
        session.scalars(
            select(AgentStepState)
            .where(AgentStepState.task_id == task_id)
            .where(AgentStepState.step_type == "workflow_plan_stage")
            .order_by(AgentStepState.sequence_index.asc(), AgentStepState.created_at.asc(), AgentStepState.id.asc())
        ).all()
    )


def _current_plan_stage(stage_steps: list[AgentStepState]) -> AgentStepState | None:
    for step in stage_steps:
        if _enum_value(step.status) in {AgentStepStatus.PENDING.value, AgentStepStatus.RUNNING.value, AgentStepStatus.WAITING_USER.value}:
            return step
    return stage_steps[-1] if stage_steps else None


def _agent_task_plan_stage_read(step: AgentStepState) -> AgentTaskPlanStageRead:
    payload = dict(step.input_payload or {})
    output_payload = dict(step.output_payload or {})
    return AgentTaskPlanStageRead(
        step_id=step.id,
        stage_id=str(payload.get("stage_id") or step.capability),
        sequence_index=step.sequence_index,
        title=str(payload.get("title") or step.capability),
        objective=str(payload.get("objective") or ""),
        business_action=str(payload.get("business_action") or "") or None,
        allowed_capabilities=[str(item) for item in payload.get("allowed_capabilities") or []],
        tool_strategy=dict(payload.get("tool_strategy") or {}),
        ranking_policy=[str(item) for item in payload.get("ranking_policy") or []],
        capability=step.capability,
        status=_enum_value(step.status),
        execution_status=str(output_payload.get("execution_status") or "") or None,
        waiting_message=str(output_payload.get("waiting_message") or "") or None,
        final_answer_preview=str(output_payload.get("final_answer_preview") or "") or None,
        depends_on=[str(item) for item in payload.get("depends_on") or []],
        received_context=dict(payload.get("received_context")) if isinstance(payload.get("received_context"), dict) else None,
        handoff_payload=dict(output_payload.get("handoff_payload")) if isinstance(output_payload.get("handoff_payload"), dict) else None,
    )


def _agent_task_read(task: AgentTaskState, *, steps: list[AgentStepState] | None = None) -> AgentTaskRead:
    return AgentTaskRead(
        id=task.id,
        root_workflow_run_id=task.root_workflow_run_id,
        conversation_session_id=task.conversation_session_id,
        task_type=task.task_type,
        capability=task.capability,
        status=_enum_value(task.status),
        current_step_id=task.current_step_id,
        owner_executor=task.owner_executor,
        user_goal=task.user_goal,
        input_payload=dict(task.input_payload or {}),
        output_payload=dict(task.output_payload or {}),
        created_at=task.created_at,
        updated_at=task.updated_at,
        completed_at=task.completed_at,
        steps=[_agent_task_step_read(step) for step in (steps or [])],
    )


def _agent_task_step_read(step: AgentStepState) -> AgentTaskStepRead:
    return AgentTaskStepRead(
        id=step.id,
        task_id=step.task_id,
        parent_step_id=step.parent_step_id,
        sequence_index=step.sequence_index,
        step_type=step.step_type,
        status=_enum_value(step.status),
        executor_type=step.executor_type,
        executor_name=step.executor_name,
        capability=step.capability,
        input_payload=dict(step.input_payload or {}),
        output_payload=dict(step.output_payload or {}),
        tool_call_log_id=step.tool_call_log_id,
        external_task_id=step.external_task_id,
        approval_request_id=step.approval_request_id,
        retry_count=step.retry_count,
        started_at=step.started_at,
        completed_at=step.completed_at,
        created_at=step.created_at,
        updated_at=step.updated_at,
    )


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


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
