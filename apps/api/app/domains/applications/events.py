from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.domains.applications.models import ApplicationStatus


@dataclass(frozen=True)
class ApplicationCreated:
    application_id: str
    job_id: str
    status: ApplicationStatus
    occurred_at: datetime

    event_type: str = "ApplicationCreated"


@dataclass(frozen=True)
class ApplicationStatusChanged:
    application_id: str
    from_status: ApplicationStatus
    to_status: ApplicationStatus
    occurred_at: datetime

    event_type: str = "ApplicationStatusChanged"
