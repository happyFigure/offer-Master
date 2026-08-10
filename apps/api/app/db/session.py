from collections.abc import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings


def create_engine_from_url(database_url: str | URL) -> Engine:
    return create_engine(database_url, future=True, pool_pre_ping=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )


def create_session_factory_from_settings(
    settings: Settings | None = None,
) -> sessionmaker[Session]:
    runtime_settings = settings or get_settings()
    engine = create_engine_from_url(runtime_settings.database_url)
    return create_session_factory(engine)


SessionLocal = create_session_factory_from_settings()


def get_db_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
