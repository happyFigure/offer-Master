from __future__ import annotations

from dataclasses import dataclass

from app.domains.applications.events import ApplicationCreated
from app.domains.applications.models import Application, ApplicationEvent, utc_now
from app.domains.applications.repository import (
    ApplicationEventRepository,
    ApplicationRepository,
)
from app.domains.applications.schemas import ApplicationCreate, ApplicationUpdate


@dataclass(frozen=True)
class ApplicationCreateResult:
    application: Application
    timeline_event: ApplicationEvent
    event: ApplicationCreated


class ApplicationService:
    def __init__(
        self,
        applications: ApplicationRepository,
        events: ApplicationEventRepository,
    ) -> None:
        self._applications = applications
        self._events = events

    def create_application(self, command: ApplicationCreate) -> ApplicationCreateResult:
        application = self._applications.add(
            Application(
                job_id=command.job_id,
                status=command.status,
                priority=command.priority,
                channel=command.channel,
                applied_at=command.applied_at,
                next_follow_up_at=command.next_follow_up_at,
                notes=command.notes,
            )
        )
        timeline_event = self._events.add(
            ApplicationEvent(
                application=application,
                event_type="application_created",
                from_status=None,
                to_status=application.status,
                title="Application created",
                body=application.notes,
                actor="user",
                source="domain",
                event_metadata={"priority": application.priority},
            )
        )
        return ApplicationCreateResult(
            application=application,
            timeline_event=timeline_event,
            event=ApplicationCreated(
                application_id=application.id,
                job_id=application.job_id,
                status=application.status,
                occurred_at=utc_now(),
            ),
        )

    def list_applications(self, limit: int = 100):
        return self._applications.list_filtered(limit=limit)

    def update_application(self, application_id: str, command: ApplicationUpdate) -> Application:
        application = self._applications.get(application_id)
        if application is None:
            raise ValueError(f"Application not found: {application_id}")

        previous_status = application.status
        updates = command.model_dump(exclude_unset=True, exclude={"actor", "source"})
        for field, value in updates.items():
            setattr(application, field, value)

        if command.status is not None and command.status != previous_status:
            self._events.add(
                ApplicationEvent(
                    application=application,
                    event_type="status_changed",
                    from_status=previous_status,
                    to_status=command.status,
                    title="Application status changed",
                    body=command.notes,
                    actor=command.actor,
                    source=command.source,
                    event_metadata={"priority": application.priority},
                )
            )
        elif updates:
            self._events.add(
                ApplicationEvent(
                    application=application,
                    event_type="application_updated",
                    from_status=previous_status,
                    to_status=application.status,
                    title="Application updated",
                    body=command.notes,
                    actor=command.actor,
                    source=command.source,
                    event_metadata={"updated_fields": sorted(updates)},
                )
            )
        return application
