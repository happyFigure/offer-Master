from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.domains.jobs.models import JobLeadStatus
from app.domains.jobs.repository import (
    CompanyRepository,
    JobLeadRepository,
    JobRepository,
    JobSourceRepository,
    RawJobLeadRepository,
    SourceSyncRunRepository,
)
from app.domains.jobs.schemas import (
    CompanySummaryRead,
    JobLeadConversionResponse,
    JobLeadCreate,
    JobLeadListResponse,
    JobLeadRead,
    JobLeadVerification,
    JobSourceCreate,
    JobSourceListResponse,
    JobSourceRead,
    JobSummaryRead,
    RawJobLeadCaptureResponse,
    RawJobLeadCreate,
)
from app.domains.jobs.service import JobLeadService, JobService


source_router = APIRouter(prefix="/api/v1/job-sources", tags=["job-sources"])
lead_router = APIRouter(prefix="/api/v1/job-leads", tags=["job-leads"])


@source_router.post("", response_model=JobSourceRead, status_code=status.HTTP_201_CREATED)
def create_job_source(
    request: JobSourceCreate,
    session: Session = Depends(get_db_session),
) -> JobSourceRead:
    service = _lead_service(session)
    source = service.create_source(request)
    session.commit()
    return JobSourceRead.model_validate(source)


@source_router.get("", response_model=JobSourceListResponse)
def list_job_sources(session: Session = Depends(get_db_session)) -> JobSourceListResponse:
    sources = JobSourceRepository(session).list_enabled()
    return JobSourceListResponse(items=[JobSourceRead.model_validate(source) for source in sources])


@lead_router.post("/raw", response_model=RawJobLeadCaptureResponse, status_code=status.HTTP_201_CREATED)
def capture_raw_job_lead(
    request: RawJobLeadCreate,
    session: Session = Depends(get_db_session),
) -> RawJobLeadCaptureResponse:
    try:
        result = _lead_service(session).capture_raw_lead(request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    return RawJobLeadCaptureResponse(
        raw_lead=result.raw_lead,
        created=result.created,
    )


@lead_router.post("", response_model=JobLeadRead, status_code=status.HTTP_201_CREATED)
def create_job_lead(
    request: JobLeadCreate,
    session: Session = Depends(get_db_session),
) -> JobLeadRead:
    try:
        lead = _lead_service(session).create_lead(request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    return JobLeadRead.model_validate(lead)


@lead_router.get("", response_model=JobLeadListResponse)
def list_job_leads(
    verification_status: JobLeadStatus | None = None,
    session: Session = Depends(get_db_session),
) -> JobLeadListResponse:
    leads = JobLeadRepository(session).list_by_status(verification_status)
    return JobLeadListResponse(items=[JobLeadRead.model_validate(lead) for lead in leads])


@lead_router.post("/{lead_id}/verify", response_model=JobLeadRead)
def verify_job_lead(
    lead_id: str,
    request: JobLeadVerification,
    session: Session = Depends(get_db_session),
) -> JobLeadRead:
    try:
        lead = _lead_service(session).verify_lead(lead_id, request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    return JobLeadRead.model_validate(lead)


@lead_router.post("/{lead_id}/convert", response_model=JobLeadConversionResponse)
def convert_job_lead(
    lead_id: str,
    session: Session = Depends(get_db_session),
) -> JobLeadConversionResponse:
    lead_service = _lead_service(session)
    job_service = JobService(
        companies=CompanyRepository(session),
        jobs=JobRepository(session),
    )
    try:
        result = lead_service.convert_verified_lead_to_job(lead_id, job_service)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.commit()
    return JobLeadConversionResponse(
        lead=JobLeadRead.model_validate(result.lead),
        job=_job_summary(result.job),
        created=result.created,
    )


def _lead_service(session: Session) -> JobLeadService:
    return JobLeadService(
        sources=JobSourceRepository(session),
        sync_runs=SourceSyncRunRepository(session),
        raw_leads=RawJobLeadRepository(session),
        leads=JobLeadRepository(session),
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
