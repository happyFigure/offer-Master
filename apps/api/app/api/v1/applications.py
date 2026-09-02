from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.domains.applications.repository import ApplicationEventRepository, ApplicationRepository
from app.domains.applications.schemas import (
    ApplicationBoardItem,
    ApplicationCreate,
    ApplicationFromJobCreate,
    ApplicationListResponse,
    ApplicationRead,
    ApplicationUpdate,
)
from app.domains.applications.service import ApplicationService
from app.domains.jobs.repository import CompanyRepository, JobRepository
from app.domains.jobs.schemas import CompanySummaryRead, JobSummaryRead
from app.domains.jobs.service import JobService


router = APIRouter(prefix="/api/v1/applications", tags=["applications"])


@router.get("", response_model=ApplicationListResponse)
def list_applications(
    limit: int = Query(default=100, ge=1, le=200),
    session: Session = Depends(get_db_session),
) -> ApplicationListResponse:
    service = _application_service(session)
    return ApplicationListResponse(items=[_application_board_item(item) for item in service.list_applications(limit=limit)])


@router.post("", response_model=ApplicationBoardItem)
def create_application(
    request: ApplicationCreate,
    session: Session = Depends(get_db_session),
) -> ApplicationBoardItem:
    service = _application_service(session)
    try:
        result = service.create_application(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.commit()
    return _application_board_item(result.application)


@router.post("/from-job", response_model=ApplicationBoardItem)
def create_application_from_job(
    request: ApplicationFromJobCreate,
    session: Session = Depends(get_db_session),
) -> ApplicationBoardItem:
    job_service = JobService(companies=CompanyRepository(session), jobs=JobRepository(session))
    application_repository = ApplicationRepository(session)
    service = ApplicationService(
        applications=application_repository,
        events=ApplicationEventRepository(session),
    )

    try:
        imported = job_service.import_job(request.job)
        existing = application_repository.list_by_job(imported.job.id)
        if existing:
            session.commit()
            return _application_board_item(existing[0])

        result = service.create_application(
            ApplicationCreate(
                job_id=imported.job.id,
                status=request.status,
                priority=request.priority,
                channel=request.channel,
                applied_at=request.applied_at,
                next_follow_up_at=request.next_follow_up_at,
                notes=request.notes,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.commit()
    return _application_board_item(result.application)


@router.patch("/{application_id}", response_model=ApplicationBoardItem)
def update_application(
    application_id: str,
    request: ApplicationUpdate,
    session: Session = Depends(get_db_session),
) -> ApplicationBoardItem:
    service = _application_service(session)
    try:
        application = service.update_application(application_id, request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.commit()
    return _application_board_item(application)


def _application_service(session: Session) -> ApplicationService:
    return ApplicationService(
        applications=ApplicationRepository(session),
        events=ApplicationEventRepository(session),
    )


def _application_board_item(application) -> ApplicationBoardItem:
    return ApplicationBoardItem(
        id=application.id,
        job_id=application.job_id,
        status=application.status,
        priority=application.priority,
        channel=application.channel,
        applied_at=application.applied_at,
        next_follow_up_at=application.next_follow_up_at,
        notes=application.notes,
        created_at=application.created_at,
        updated_at=application.updated_at,
        job=_job_summary(application.job),
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
