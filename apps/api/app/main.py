from fastapi import FastAPI

from app.api.v1.health import router as health_router


def create_app() -> FastAPI:
    app = FastAPI(title="JobPilot API", version="0.1.0")
    app.include_router(health_router)
    return app


app = create_app()

