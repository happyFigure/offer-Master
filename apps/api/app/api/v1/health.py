from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    return {
        "service": "jobpilot-api",
        "status": "ok",
        "architecture": "ddd-langgraph-modular-monolith",
    }

