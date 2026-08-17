from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.agent_runtime.memory.memory_tools import memory_get, memory_search, sessions_history, sessions_search
from app.db.session import get_db_session


router = APIRouter(prefix="/api/v1/agent-memory", tags=["agent-memory"])


@router.get("/search")
def search_agent_memory(
    query: str,
    corpus: str | None = None,
    limit: int = Query(default=10, ge=1, le=50),
    session: Session = Depends(get_db_session),
):
    return asdict(memory_search(session, query=query, corpus=corpus, limit=limit))


@router.get("/{memory_id}")
def get_agent_memory(memory_id: str, session: Session = Depends(get_db_session)):
    return asdict(memory_get(session, memory_id=memory_id))


session_recall_router = APIRouter(prefix="/api/v1/agent", tags=["agent-session-recall"])


@session_recall_router.get("/sessions/search")
def search_agent_sessions(
    query: str,
    limit: int = Query(default=10, ge=1, le=50),
    session: Session = Depends(get_db_session),
):
    return asdict(sessions_search(session, query=query, limit=limit))


@session_recall_router.get("/sessions/{session_key}/history")
def get_agent_session_history(
    session_key: str,
    around_message_id: str | None = None,
    window_before: int = Query(default=5, ge=0, le=50),
    window_after: int = Query(default=5, ge=0, le=50),
    session: Session = Depends(get_db_session),
):
    try:
        return asdict(
            sessions_history(
                session,
                session_key=session_key,
                around_message_id=around_message_id,
                window_before=window_before,
                window_after=window_after,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
