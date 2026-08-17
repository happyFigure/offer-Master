from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.agent_runtime.memory.skill_dependency_gate import SkillDependencyGate
from app.agent_runtime.memory.skill_repository import AgentSkillRepository
from app.agent_runtime.tool_registry import AgentToolRegistry, create_default_agent_tool_registry
from app.db.session import get_db_session
from app.domains.agent_memory.models import AgentSkill, AgentSkillStatus
from app.domains.agent_memory.repository import AgentMemoryRepository
from app.domains.agent_memory.schemas import (
    AgentSkillDocumentRead,
    AgentSkillImportRequest,
    AgentSkillListResponse,
    AgentSkillRead,
    AgentSkillUsageEventRequest,
    AgentSkillUsageRead,
)


router = APIRouter(prefix="/api/v1/agent-skills", tags=["agent-skills"])


def get_skill_repository(session: Session = Depends(get_db_session)) -> AgentSkillRepository:
    return AgentSkillRepository(AgentMemoryRepository(session))


def get_agent_tool_registry() -> AgentToolRegistry:
    return create_default_agent_tool_registry()


@router.get("", response_model=AgentSkillListResponse)
def list_agent_skills(
    status: AgentSkillStatus | None = None,
    limit: int = Query(default=100, ge=1, le=200),
    repository: AgentSkillRepository = Depends(get_skill_repository),
    tool_registry: AgentToolRegistry = Depends(get_agent_tool_registry),
) -> AgentSkillListResponse:
    repository.ensure_builtin_content_source_skills()
    skills = repository.list_skills(status=status, limit=limit)
    _decorate_skill_dependencies(skills, tool_registry)
    return AgentSkillListResponse(items=[AgentSkillRead.model_validate(skill) for skill in skills])


@router.post("/import", response_model=AgentSkillRead, status_code=status.HTTP_201_CREATED)
def import_agent_skill(
    request: AgentSkillImportRequest,
    session: Session = Depends(get_db_session),
    repository: AgentSkillRepository = Depends(get_skill_repository),
    tool_registry: AgentToolRegistry = Depends(get_agent_tool_registry),
) -> AgentSkillRead:
    try:
        skill = repository.import_skill_from_path(
            request.source_path,
            category=request.category,
            protected=request.protected,
            pinned=request.pinned,
            created_by=request.created_by,
            metadata_json=request.metadata_json,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _decorate_skill_dependencies([skill], tool_registry)
    session.commit()
    return AgentSkillRead.model_validate(skill)


@router.get("/{skill_id}", response_model=AgentSkillDocumentRead)
def get_agent_skill(
    skill_id: str,
    repository: AgentSkillRepository = Depends(get_skill_repository),
    tool_registry: AgentToolRegistry = Depends(get_agent_tool_registry),
) -> AgentSkillDocumentRead:
    try:
        document = repository.read_skill(skill_id, record_view=True)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _decorate_skill_dependencies([document.skill], tool_registry)
    return AgentSkillDocumentRead(
        skill=AgentSkillRead.model_validate(document.skill),
        content=document.content,
        version_hash=document.version_hash,
    )


def _decorate_skill_dependencies(skills: list[AgentSkill], tool_registry: AgentToolRegistry) -> None:
    gate = SkillDependencyGate(tool_registry)
    for skill in skills:
        skill.metadata_json = gate.decorate_metadata(skill.metadata_json)


@router.post("/{skill_id}/usage", response_model=AgentSkillUsageRead)
def record_agent_skill_usage(
    skill_id: str,
    request: AgentSkillUsageEventRequest,
    session: Session = Depends(get_db_session),
    repository: AgentSkillRepository = Depends(get_skill_repository),
) -> AgentSkillUsageRead:
    try:
        usage = repository.record_usage(skill_id, request.event)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    return AgentSkillUsageRead.model_validate(usage)


@router.post("/{skill_id}/pin", response_model=AgentSkillRead)
def pin_agent_skill(
    skill_id: str,
    session: Session = Depends(get_db_session),
    repository: AgentSkillRepository = Depends(get_skill_repository),
) -> AgentSkillRead:
    try:
        skill = repository.pin_skill(skill_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    return AgentSkillRead.model_validate(skill)


@router.post("/{skill_id}/archive", response_model=AgentSkillRead)
def archive_agent_skill(
    skill_id: str,
    session: Session = Depends(get_db_session),
    repository: AgentSkillRepository = Depends(get_skill_repository),
) -> AgentSkillRead:
    try:
        skill = repository.archive_skill(skill_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    return AgentSkillRead.model_validate(skill)
