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
