from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domains.applications.models import Application, ApplicationEvent, ApplicationStatus


class ApplicationRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, application_id: str) -> Application | None:
        return self._session.get(Application, application_id)

    def list_by_job(self, job_id: str) -> list[Application]:
        return list(
            self._session.scalars(
                select(Application)
                .where(Application.job_id == job_id)
                .order_by(Application.created_at.desc())
            ).all()
        )

    def list_by_status(self, status: ApplicationStatus) -> list[Application]:
        return list(
            self._session.scalars(
                select(Application)
                .where(Application.status == status)
                .order_by(Application.created_at.desc())
            ).all()
        )

    def list_filtered(self, status: ApplicationStatus | None = None, limit: int = 100) -> list[Application]:
        statement = select(Application).order_by(Application.updated_at.desc())
        if status is not None:
            statement = statement.where(Application.status == status)
        return list(self._session.scalars(statement.limit(limit)).all())

    def add(self, application: Application) -> Application:
        self._session.add(application)
        self._session.flush()
        return application


class ApplicationEventRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def list_for_application(self, application_id: str) -> list[ApplicationEvent]:
        return list(
            self._session.scalars(
                select(ApplicationEvent)
                .where(ApplicationEvent.application_id == application_id)
                .order_by(ApplicationEvent.created_at)
            ).all()
        )

    def add(self, event: ApplicationEvent) -> ApplicationEvent:
        self._session.add(event)
        self._session.flush()
        return event
