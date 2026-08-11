from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class JobImported:
    job_id: str
    company_id: str
    source: str
    source_job_id: str
    occurred_at: datetime

    event_type: str = "JobImported"


@dataclass(frozen=True)
class JobLeadCaptured:
    raw_lead_id: str
    source_id: str
    content_hash: str
    occurred_at: datetime

    event_type: str = "JobLeadCaptured"


@dataclass(frozen=True)
class JobLeadCreated:
    lead_id: str
    source_id: str
    raw_lead_id: str | None
    occurred_at: datetime

    event_type: str = "JobLeadCreated"


@dataclass(frozen=True)
class JobLeadVerified:
    lead_id: str
    source_id: str
    verification_status: str
    occurred_at: datetime

    event_type: str = "JobLeadVerified"


@dataclass(frozen=True)
class JobLeadConverted:
    lead_id: str
    job_id: str
    source_id: str
    occurred_at: datetime

    event_type: str = "JobLeadConverted"
