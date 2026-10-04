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
    ApplicationMailNotificationConfirm,
    ApplicationMailNotificationCreate,
    ApplicationMailNotificationUpdate,
    ApplicationMailTimingRead,
    ApplicationNotificationListResponse,
    ApplicationNotificationRead,
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


@router.get("/notification-candidates", response_model=ApplicationNotificationListResponse)
def list_notification_candidates(
    limit: int = Query(default=100, ge=1, le=200),
    session: Session = Depends(get_db_session),
) -> ApplicationNotificationListResponse:
    service = _application_service(session)
    items = service.list_pending_mail_notifications(limit=limit)
    return ApplicationNotificationListResponse(items=[_notification_read(item) for item in items])


@router.post("/{application_id}/notification-candidates", response_model=ApplicationNotificationRead)
def create_notification_candidate(
    application_id: str,
    request: ApplicationMailNotificationCreate,
    session: Session = Depends(get_db_session),
) -> ApplicationNotificationRead:
    if request.application_id != application_id:
        raise HTTPException(status_code=400, detail="application_id in path and body must match")
    service = _application_service(session)
    try:
        event = service.create_mail_notification_candidate(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.commit()
    return _notification_read(event)


@router.patch("/notification-candidates/{event_id}", response_model=ApplicationNotificationRead)
def update_notification_candidate(
    event_id: str,
    request: ApplicationMailNotificationUpdate,
    session: Session = Depends(get_db_session),
) -> ApplicationNotificationRead:
    service = _application_service(session)
    try:
        event = service.update_mail_notification_candidate(event_id, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.commit()
    return _notification_read(event)


@router.post("/notification-candidates/{event_id}/confirm", response_model=ApplicationNotificationRead)
def confirm_notification_candidate(
    event_id: str,
    request: ApplicationMailNotificationConfirm | None = None,
    session: Session = Depends(get_db_session),
) -> ApplicationNotificationRead:
    service = _application_service(session)
    try:
        service.confirm_mail_notification_candidate(event_id, (request or ApplicationMailNotificationConfirm()).to_status)
        event = service.get_mail_notification_candidate(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Mail notification candidate not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.commit()
    return _notification_read(event)


@router.post("/notification-candidates/{event_id}/reject", response_model=ApplicationNotificationRead)
def reject_notification_candidate(
    event_id: str,
    session: Session = Depends(get_db_session),
) -> ApplicationNotificationRead:
    service = _application_service(session)
    try:
        event = service.reject_mail_notification_candidate(event_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.commit()
    return _notification_read(event)


def _application_service(session: Session) -> ApplicationService:
    return ApplicationService(
        applications=ApplicationRepository(session),
        events=ApplicationEventRepository(session),
    )


def _application_board_item(application) -> ApplicationBoardItem:
    latest_mail_event = _latest_mail_event(application)
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
        mail_timing=_mail_timing(latest_mail_event) if latest_mail_event is not None else None,
    )


def _latest_mail_event(application):
    events = [
        event
        for event in getattr(application, "events", [])
        if event.source == "qq_mail" and event.review_status in {"pending", "confirmed", "synced"}
    ]
    return max(events, key=lambda event: event.created_at) if events else None


def _mail_timing(event) -> ApplicationMailTimingRead:
    metadata = event.event_metadata or {}
    return ApplicationMailTimingRead(
        event_type=event.event_type,
        title=event.title,
        source_sent_at=metadata.get("source_sent_at"),
        scheduled_at=event.scheduled_at,
        deadline_at=event.deadline_at,
        deadline_offset_hours=metadata.get("deadline_offset_hours"),
        timezone=event.timezone,
        timing_source=str(metadata.get("timing_source") or "none"),
        timing_note=metadata.get("timing_note"),
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


def _notification_read(event) -> ApplicationNotificationRead:
    return ApplicationNotificationRead(
        id=event.id,
        application_id=event.application_id,
        event_type=event.event_type,
        from_status=event.from_status,
        to_status=event.to_status,
        title=event.title,
        body=event.body,
        actor=event.actor,
        source=event.source,
        event_metadata=event.event_metadata,
        scheduled_at=event.scheduled_at,
        deadline_at=event.deadline_at,
        timezone=event.timezone,
        join_url=event.join_url,
        source_message_id=event.source_message_id,
        source_uid=event.source_uid,
        review_status=event.review_status,
        reviewed_at=event.reviewed_at,
        created_at=event.created_at,
        application=_application_board_item(event.application),
    )
