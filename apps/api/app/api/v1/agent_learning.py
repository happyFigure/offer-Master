from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.agent_runtime.memory.skill_repository import AgentSkillRepository
from app.api.v1.agent_skills import get_skill_repository
from app.db.session import get_db_session
from app.domains.agent_memory.models import AgentLearningCandidateStatus
from app.domains.agent_memory.repository import AgentMemoryRepository
from app.domains.agent_memory.schemas import AgentLearningCandidateListResponse, AgentLearningCandidateRead
from app.domains.agent_memory.service import AgentLearningService


router = APIRouter(prefix="/api/v1/agent-learning", tags=["agent-learning"])


@router.get("/candidates", response_model=AgentLearningCandidateListResponse)
def list_learning_candidates(
    status: AgentLearningCandidateStatus | None = None,
    limit: int = Query(default=100, ge=1, le=200),
    session: Session = Depends(get_db_session),
) -> AgentLearningCandidateListResponse:
    candidates = _learning_service(session).list_learning_candidates(status=status, limit=limit)
    return AgentLearningCandidateListResponse(
        items=[AgentLearningCandidateRead.model_validate(candidate) for candidate in candidates]
    )


@router.post("/candidates/{candidate_id}/approve", response_model=AgentLearningCandidateRead)
def approve_learning_candidate(candidate_id: str, session: Session = Depends(get_db_session)) -> AgentLearningCandidateRead:
    try:
        candidate = _learning_service(session).approve_candidate(candidate_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    return AgentLearningCandidateRead.model_validate(candidate)


@router.post("/candidates/{candidate_id}/reject", response_model=AgentLearningCandidateRead)
def reject_learning_candidate(candidate_id: str, session: Session = Depends(get_db_session)) -> AgentLearningCandidateRead:
    try:
        candidate = _learning_service(session).reject_candidate(candidate_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    return AgentLearningCandidateRead.model_validate(candidate)


@router.post("/candidates/{candidate_id}/apply", response_model=AgentLearningCandidateRead)
def apply_learning_candidate(
    candidate_id: str,
    session: Session = Depends(get_db_session),
    skill_repository: AgentSkillRepository = Depends(get_skill_repository),
) -> AgentLearningCandidateRead:
    try:
        candidate = _learning_service(session).apply_candidate(candidate_id, skill_repository=skill_repository)
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if "not found" in detail else 409
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    session.commit()
    return AgentLearningCandidateRead.model_validate(candidate)


def _learning_service(session: Session) -> AgentLearningService:
    return AgentLearningService(AgentMemoryRepository(session))
