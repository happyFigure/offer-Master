from __future__ import annotations

from dataclasses import dataclass

from app.domains.applications.events import ApplicationCreated
from app.domains.applications.models import Application, ApplicationEvent, utc_now
from app.domains.applications.repository import (
    ApplicationEventRepository,
    ApplicationRepository,
)
from app.domains.applications.schemas import ApplicationCreate


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
