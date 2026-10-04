from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domains.applications.models import Application, ApplicationEvent, ApplicationStatus
from app.domains.jobs.models import Company, Job


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

    def list_for_mail_match(
        self,
        *,
        company_name: str | None = None,
        job_title: str | None = None,
        limit: int = 20,
    ) -> list[Application]:
        """Find existing applications for model-assisted mail matching.

        This is deliberately read-only and returns only the fields needed to
        choose an existing application before a mail event is staged.
        """
        statement = (
            select(Application)
            .join(Application.job)
            .join(Job.company)
            .order_by(Application.updated_at.desc())
        )
        filters = []
        if company_name:
            filters.append(Company.name.ilike(f"%{company_name.strip()}%"))
        if job_title:
            filters.append(Job.title.ilike(f"%{job_title.strip()}%"))
        if filters:
            from sqlalchemy import or_

            statement = statement.where(or_(*filters))
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

    def list_pending_mail_notifications(self, limit: int = 100) -> list[ApplicationEvent]:
        return list(
            self._session.scalars(
                select(ApplicationEvent)
                .where(ApplicationEvent.source == "qq_mail")
                .where(ApplicationEvent.review_status == "pending")
                .order_by(ApplicationEvent.deadline_at.asc(), ApplicationEvent.created_at.desc())
                .limit(limit)
            ).all()
        )

    def get(self, event_id: str) -> ApplicationEvent | None:
        return self._session.get(ApplicationEvent, event_id)

    def get_by_mail_source(self, application_id: str, source_message_id: str, event_type: str) -> ApplicationEvent | None:
        return self._session.scalar(
            select(ApplicationEvent)
            .where(ApplicationEvent.application_id == application_id)
            .where(ApplicationEvent.source_message_id == source_message_id)
            .where(ApplicationEvent.event_type == event_type)
        )

    def get_by_mail_source_identity(self, source_message_id: str, event_type: str) -> ApplicationEvent | None:
        return self._session.scalar(
            select(ApplicationEvent)
            .where(ApplicationEvent.source == "qq_mail")
            .where(ApplicationEvent.source_message_id == source_message_id)
            .where(ApplicationEvent.event_type == event_type)
        )

    def add(self, event: ApplicationEvent) -> ApplicationEvent:
        self._session.add(event)
        self._session.flush()
        return event
