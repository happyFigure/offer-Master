from fastapi import FastAPI

from app.api.v1.agent import router as agent_router
from app.api.v1.agent_learning import router as agent_learning_router
from app.api.v1.agent_memory import router as agent_memory_router, session_recall_router
from app.api.v1.agent_runtime import router as agent_runtime_router
from app.api.v1.agent_skills import router as agent_skills_router
from app.api.v1.applications import router as applications_router
from app.api.v1.health import router as health_router
from app.api.v1.job_sources import (
    article_candidate_router,
    lead_router,
    recruiting_signal_router,
    source_router,
    tool_health_router,
)
from app.api.v1.jobs import router as jobs_router


def create_app() -> FastAPI:
    app = FastAPI(title="JobPilot API", version="0.1.0")
    app.include_router(health_router)
    app.include_router(jobs_router)
    app.include_router(applications_router)
    app.include_router(source_router)
    app.include_router(lead_router)
    app.include_router(article_candidate_router)
    app.include_router(recruiting_signal_router)
    app.include_router(tool_health_router)
    app.include_router(session_recall_router)
    app.include_router(agent_router)
    app.include_router(agent_runtime_router)
    app.include_router(agent_memory_router)
    app.include_router(agent_learning_router)
    app.include_router(agent_skills_router)
    return app


app = create_app()
