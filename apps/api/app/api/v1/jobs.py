from __future__ import annotations

from collections.abc import Mapping

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.domains.jobs.providers.base import JobProvider, JobSearchQuery
from app.domains.jobs.providers.jobicy import JobicyProvider
from app.domains.jobs.repository import CompanyRepository, JobRepository
from app.domains.jobs.schemas import (
    CompanySummaryRead,
    JobSummaryRead,
    JobSyncRequest,
    JobSyncResponse,
)
from app.domains.jobs.service import JobService


router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


def get_job_providers() -> Mapping[str, JobProvider]:
    return {"jobicy": JobicyProvider()}


@router.post("/sync", response_model=JobSyncResponse)
def sync_jobs(
    request: JobSyncRequest,
    session: Session = Depends(get_db_session),
    providers: Mapping[str, JobProvider] = Depends(get_job_providers),
) -> JobSyncResponse:
    unsupported_sources = [source for source in request.sources if source not in providers]
    if unsupported_sources:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported job providers: {', '.join(unsupported_sources)}",
        )

    service = JobService(
        companies=CompanyRepository(session),
        jobs=JobRepository(session),
    )
    query = JobSearchQuery(
        keyword=request.keyword,
        city=request.city,
        remote_type=request.remote_type,
        job_type=request.job_type,
        sources=request.sources,
        limit=request.limit,
    )

    imported = 0
    duplicates = 0
    failed = 0
    synced_jobs: list[JobSummaryRead] = []

    for source in request.sources:
        provider = providers[source]
        try:
            raw_jobs = provider.search(query)
        except Exception:
            failed += 1
            continue

        for raw_job in raw_jobs:
            try:
                result = service.import_job(provider.normalize(raw_job))
            except Exception:
                failed += 1
                continue

            if result.created:
                imported += 1
            else:
                duplicates += 1
            synced_jobs.append(_job_summary(result.job))

    session.commit()

    return JobSyncResponse(
        requested_sources=request.sources,
        imported=imported,
        duplicates=duplicates,
        failed=failed,
        jobs=synced_jobs,
    )


def _job_summary(job) -> JobSummaryRead:
    return JobSummaryRead(
        id=job.id,
        title=job.title,
        company=CompanySummaryRead(id=job.company.id, name=job.company.name),
        city=job.city,
        source=job.source,
        source_job_id=job.source_job_id,
        source_url=job.source_url,
        job_type=job.job_type,
        skills=job.skills,
        status=job.status,
    )
