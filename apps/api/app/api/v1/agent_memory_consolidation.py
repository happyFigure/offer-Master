from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.agent_runtime.memory.consolidation import MemoryConsolidationCommand, MemoryConsolidationService
from app.db.session import get_db_session
from app.domains.agent_memory.repository import AgentMemoryRepository
from app.domains.agent_memory.service import AgentLearningService
from app.domains.automation.models import WorkflowRun
from app.domains.conversations.models import AgentSession


router = APIRouter(prefix="/api/v1/agent-memory", tags=["agent-memory"])


class AgentMemoryConsolidationRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    workflow_run_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    agent_run_id: str | None = Field(default=None, min_length=1)
    target_scope: str = Field(default="agent_memory", min_length=1, max_length=128)
    message_ids: list[str] = Field(default_factory=list)


class AgentMemoryConsolidationResponse(BaseModel):
    workflow_run_id: str
    reviewed_message_count: int
    reviewed_tool_call_count: int
    created_candidate_count: int
    pending_candidate_count: int
    promoted_memory_count: int
    merged_memory_count: int
    created_candidate_ids: list[str]
    pending_candidate_ids: list[str]
    promoted_memory_ids: list[str]
    merged_memory_ids: list[str]
    skipped_reasons: list[str]


@router.post("/consolidate", response_model=AgentMemoryConsolidationResponse)
def consolidate_agent_memory(
    request: AgentMemoryConsolidationRequest,
    session: Session = Depends(get_db_session),
) -> AgentMemoryConsolidationResponse:
    if session.get(AgentSession, request.session_id) is None:
        raise HTTPException(status_code=404, detail="Agent session not found")
    if session.get(WorkflowRun, request.workflow_run_id) is None:
        raise HTTPException(status_code=404, detail="Workflow run not found")

    repository = AgentMemoryRepository(session)
    result = MemoryConsolidationService(
        session=session,
        learning_service=AgentLearningService(repository),
    ).consolidate(
        MemoryConsolidationCommand(
            session_id=request.session_id,
            workflow_run_id=request.workflow_run_id,
            agent_run_id=request.agent_run_id,
            target_scope=request.target_scope,
            message_ids=list(request.message_ids),
        )
    )
    session.commit()

    return AgentMemoryConsolidationResponse(
        workflow_run_id=result.workflow_run_id,
        reviewed_message_count=result.reviewed_message_count,
        reviewed_tool_call_count=result.reviewed_tool_call_count,
        created_candidate_count=result.created_candidate_count,
        pending_candidate_count=result.pending_candidate_count,
        promoted_memory_count=result.promoted_memory_count,
        merged_memory_count=result.merged_memory_count,
        created_candidate_ids=result.created_candidate_ids,
        pending_candidate_ids=result.pending_candidate_ids,
        promoted_memory_ids=result.promoted_memory_ids,
        merged_memory_ids=result.merged_memory_ids,
        skipped_reasons=result.skipped_reasons,
    )
