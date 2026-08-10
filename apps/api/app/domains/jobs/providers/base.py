from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.domains.jobs.schemas import JobImportDraft


@dataclass(frozen=True)
class JobSearchQuery:
    keyword: str | None = None
    city: str | None = None
    remote_type: str = "remote"
    job_type: str = "any"
    sources: list[str] = field(default_factory=list)
    limit: int = 20


@dataclass(frozen=True)
class RawJob:
    source: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ProviderStatus:
    name: str
    available: bool
    message: str | None = None


class JobProvider(Protocol):
    name: str
    source_type: str

    def health_check(self) -> ProviderStatus:
        ...

    def search(self, query: JobSearchQuery) -> list[RawJob]:
        ...

    def normalize(self, raw_job: RawJob) -> JobImportDraft:
        ...
