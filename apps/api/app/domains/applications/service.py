from __future__ import annotations

from dataclasses import dataclass
import logging
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.domains.applications.events import ApplicationCreated
from app.domains.applications.models import Application, ApplicationEvent, ApplicationStatus, utc_now
from app.domains.applications.repository import (
    ApplicationEventRepository,
    ApplicationRepository,
)
from app.domains.applications.schemas import (
    ApplicationCreate,
    ApplicationMailNotificationCreate,
    ApplicationMailSyncCreate,
    ApplicationMailNotificationUpdate,
    ApplicationUpdate,
)
from app.domains.jobs.repository import CompanyRepository, JobRepository
from app.domains.jobs.schemas import JobImportDraft
from app.domains.jobs.service import JobService


logger = logging.getLogger(__name__)


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
        jobs: JobService | None = None,
    ) -> None:
        self._applications = applications
        self._events = events
        self._jobs = jobs

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

    def list_pending_mail_notifications(self, limit: int = 100):
        return self._events.list_pending_mail_notifications(limit=limit)

    def get_mail_notification_candidate(self, event_id: str) -> ApplicationEvent | None:
        event = self._events.get(event_id)
        if event is None or event.source != "qq_mail":
            return None
        return event

    def create_mail_notification_candidate(self, command: ApplicationMailNotificationCreate) -> ApplicationEvent:
        application = self._applications.get(command.application_id)
        if application is None:
            raise ValueError(f"Application not found: {command.application_id}")
        source_message_id = _normalize_message_id(command.source_message_id)
        existing = self._events.get_by_mail_source(command.application_id, source_message_id, command.event_type)
        if existing is not None:
            logger.info(
                "Mail notification candidate deduplicated",
                extra={"application_id": command.application_id, "event_type": command.event_type},
            )
            return existing
        _validate_notification_times(command.scheduled_at, command.deadline_at)
        _validate_join_url(command.join_url)
        metadata = {
            "company_name": command.company_name,
            "job_title": command.job_title,
            "confidence": command.confidence,
            "evidence": [item[:500] for item in command.evidence],
            "to_status": command.to_status.value if command.to_status else None,
            "model_name": command.model_name,
            "parser_version": command.parser_version,
        }
        event = self._events.add(
            ApplicationEvent(
                application_id=application.id,
                event_type=command.event_type,
                title=command.title,
                body=command.body,
                actor="mail_agent",
                source="qq_mail",
                event_metadata=metadata,
                scheduled_at=command.scheduled_at,
                deadline_at=command.deadline_at,
                timezone=command.timezone,
                join_url=command.join_url,
                source_message_id=source_message_id,
                source_uid=command.source_uid,
                review_status="pending",
            )
        )
        logger.info(
            "Mail notification candidate created",
            extra={"application_id": application.id, "event_id": event.id, "event_type": command.event_type},
        )
        return event

    def sync_mail_application(self, command: ApplicationMailSyncCreate) -> tuple[Application, ApplicationEvent, bool]:
        """Create or update a board application from structured mail evidence.

        Company identity is the minimum required business fact. A missing job
        title is represented explicitly instead of being invented, while a
        repeated Message-ID/event pair returns the existing result.
        """
        company_name = command.company_name.strip()
        source_message_id = _normalize_message_id(command.source_message_id)
        existing_event = self._events.get_by_mail_source_identity(source_message_id, command.event_type)
        if existing_event is not None:
            application = self._applications.get(existing_event.application_id)
            if application is None:
                raise ValueError(f"Mail event points to missing application: {existing_event.id}")
            self._enrich_existing_mail_event_timing(existing_event, command)
            logger.info(
                "Mail application sync deduplicated",
                extra={
                    "application_id": application.id,
                    "event_id": existing_event.id,
                    "timing_enriched": bool(existing_event.event_metadata and existing_event.event_metadata.get("timing_source")),
                },
            )
            return MailApplicationSyncResult(application, existing_event, False)

        scheduled_at, deadline_at, timing_metadata = _resolve_mail_timing(command)
        _validate_notification_times(scheduled_at, deadline_at)
        _validate_join_url(command.join_url)
        job_title = _mail_job_title(command.job_title)
        target_status = command.to_status or _status_from_mail_event(command.event_type)
        application = self._find_exact_mail_application(company_name, command.job_title)
        created_application = False

        if application is None:
            if self._jobs is None:
                raise ValueError("Job service is required to sync an unmatched mail application")
            source_job_id = _mail_job_identity(company_name, command.job_title)
            imported = self._jobs.import_job(
                JobImportDraft(
                    company_name=company_name,
                    title=job_title,
                    source="qq_mail",
                    source_job_id=source_job_id,
                    source_url=command.join_url,
                    job_type="mail_notification",
                    jd_text=command.body,
                    status="open",
                    raw_payload={
                        "source_message_id": source_message_id,
                        "source_uid": command.source_uid,
                        "event_type": command.event_type,
                    },
                )
            )
            existing_for_job = self._applications.list_by_job(imported.job.id)
            if existing_for_job:
                application = existing_for_job[0]
            else:
                application = self.create_application(
                    ApplicationCreate(
                        job_id=imported.job.id,
                        status=target_status,
                        channel="qq_mail",
                        notes="由 QQ 邮箱招聘通知同步创建；岗位名称未提供时使用占位名称。",
                    )
                ).application
                created_application = True

        if application.status != target_status:
            self.update_application(
                application.id,
                ApplicationUpdate(
                    status=target_status,
                    actor="mail_agent",
                    source="qq_mail",
                    notes=f"Mail notification synchronized: {command.title}",
                ),
            )

        event = self._events.add(
            ApplicationEvent(
                application_id=application.id,
                event_type=command.event_type,
                title=command.title,
                body=command.body,
                actor="mail_agent",
                source="qq_mail",
                event_metadata={
                    "company_name": company_name,
                    "job_title": command.job_title,
                    "confidence": command.confidence,
                    "evidence": [item[:500] for item in command.evidence],
                    "to_status": target_status.value,
                    "model_name": command.model_name,
                    "parser_version": command.parser_version,
                    **timing_metadata,
                },
                scheduled_at=scheduled_at,
                deadline_at=deadline_at,
                timezone=command.timezone,
                join_url=command.join_url,
                source_message_id=source_message_id,
                source_uid=command.source_uid,
                review_status="synced",
            )
        )
        logger.info(
            "Mail application synchronized",
            extra={
                "application_id": application.id,
                "event_id": event.id,
                "company_name": company_name,
                "event_type": command.event_type,
                "target_status": target_status.value,
                "created_application": created_application,
            },
        )
        return MailApplicationSyncResult(application, event, created_application)

    def _enrich_existing_mail_event_timing(
        self,
        event: ApplicationEvent,
        command: ApplicationMailSyncCreate,
    ) -> None:
        """Backfill timing data when an already-seen mail is read again."""
        if not any(
            value is not None
            for value in (command.source_sent_at, command.scheduled_at, command.deadline_at, command.deadline_offset_hours)
        ):
            return

        scheduled_at, deadline_at, timing_metadata = _resolve_mail_timing(command)
        if command.scheduled_at is not None:
            event.scheduled_at = scheduled_at
        elif event.scheduled_at is None:
            event.scheduled_at = scheduled_at

        if command.deadline_at is not None:
            event.deadline_at = deadline_at
        elif event.deadline_at is None:
            event.deadline_at = deadline_at

        _validate_notification_times(event.scheduled_at, event.deadline_at)
        metadata = dict(event.event_metadata or {})
        for key, value in timing_metadata.items():
            if value is not None or key not in metadata:
                metadata[key] = value
        event.event_metadata = metadata

    def _find_exact_mail_application(self, company_name: str, job_title: str | None) -> Application | None:
        normalized_company = _normalize_text(company_name)
        normalized_title = _normalize_text(job_title) if job_title else None
        for application in self._applications.list_for_mail_match(company_name=company_name, job_title=job_title):
            if _normalize_text(application.job.company.name) != normalized_company:
                continue
            if normalized_title and _normalize_text(application.job.title) != normalized_title:
                continue
            if not normalized_title and application.job.source != "qq_mail":
                continue
            return application
        return None

    def update_mail_notification_candidate(
        self,
        event_id: str,
        command: ApplicationMailNotificationUpdate,
    ) -> ApplicationEvent:
        event = self._events.get(event_id)
        if event is None or event.source != "qq_mail":
            raise ValueError(f"Mail notification candidate not found: {event_id}")
        if event.review_status != "pending":
            raise ValueError("Only pending mail notification candidates can be edited")
        updates = command.model_dump(exclude_unset=True)
        metadata = dict(event.event_metadata or {})
        for key in ("company_name", "job_title", "to_status", "evidence"):
            if key in updates:
                value = updates.pop(key)
                metadata[key] = value.value if isinstance(value, ApplicationStatus) else value
        for field, value in updates.items():
            setattr(event, field, value)
        _validate_notification_times(event.scheduled_at, event.deadline_at)
        _validate_join_url(event.join_url)
        event.event_metadata = metadata
        return event

    def reject_mail_notification_candidate(self, event_id: str) -> ApplicationEvent:
        event = self._candidate_for_action(event_id)
        if event.review_status == "confirmed":
            raise ValueError("Confirmed mail notification candidates cannot be rejected")
        event.review_status = "rejected"
        event.reviewed_at = _utc_now()
        logger.info("Mail notification candidate rejected", extra={"event_id": event.id})
        return event

    def confirm_mail_notification_candidate(
        self,
        event_id: str,
        to_status: ApplicationStatus | None = None,
    ) -> Application:
        event = self._candidate_for_action(event_id)
        application = self._applications.get(event.application_id)
        if application is None:
            raise ValueError(f"Application not found: {event.application_id}")
        if event.review_status == "confirmed":
            return application
        target_status = to_status or _status_from_metadata(event.event_metadata)
        event.review_status = "confirmed"
        event.reviewed_at = _utc_now()
        if target_status is not None and target_status != application.status:
            self.update_application(
                application.id,
                ApplicationUpdate(
                    status=target_status,
                    actor="user",
                    source="qq_mail_confirmation",
                    notes=f"Confirmed mail notification: {event.title}",
                ),
            )
        logger.info(
            "Mail notification candidate confirmed",
            extra={"event_id": event.id, "application_id": application.id, "to_status": target_status.value if target_status else None},
        )
        return application

    def _candidate_for_action(self, event_id: str) -> ApplicationEvent:
        event = self._events.get(event_id)
        if event is None or event.source != "qq_mail" or event.review_status not in {"pending", "confirmed"}:
            raise ValueError(f"Mail notification candidate not found: {event_id}")
        if event.review_status == "rejected":
            raise ValueError("Rejected mail notification candidates cannot be confirmed")
        return event

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


def _normalize_message_id(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("source_message_id is required")
    return normalized.strip("<>").join(("<", ">")) if not normalized.startswith("<") else normalized


def _validate_notification_times(scheduled_at: datetime | None, deadline_at: datetime | None) -> None:
    if scheduled_at is not None and deadline_at is not None and deadline_at < scheduled_at:
        raise ValueError("deadline_at cannot be earlier than scheduled_at")


def _resolve_mail_timing(command: ApplicationMailSyncCreate) -> tuple[datetime | None, datetime | None, dict[str, object]]:
    """Resolve explicit mail times and relative completion windows.

    The model extracts the meaning of the email. Runtime only normalizes the
    timezone and performs the arithmetic, so a relative deadline is auditable
    instead of being silently guessed from unstructured text.
    """
    timezone = (command.timezone or "Asia/Shanghai").strip()
    zone = _mail_timezone(timezone)
    source_sent_at = _normalize_mail_datetime(command.source_sent_at, zone)
    scheduled_at = _normalize_mail_datetime(command.scheduled_at, zone)
    explicit_deadline = _normalize_mail_datetime(command.deadline_at, zone)
    deadline_at = explicit_deadline
    timing_source = "explicit" if scheduled_at is not None or explicit_deadline is not None else "none"
    timing_note = None

    if deadline_at is None and command.deadline_offset_hours is not None:
        if source_sent_at is None:
            raise ValueError("source_sent_at is required when deadline_offset_hours is provided")
        deadline_at = source_sent_at + timedelta(hours=command.deadline_offset_hours)
        timing_source = "relative"
        timing_note = "截止时间按邮件发送时间加相对时限计算"

    metadata: dict[str, object] = {
        "source_sent_at": source_sent_at.isoformat() if source_sent_at else None,
        "deadline_offset_hours": command.deadline_offset_hours,
        "timing_source": timing_source,
        "timing_note": timing_note,
    }
    return scheduled_at, deadline_at, metadata


def _mail_timezone(timezone: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Unsupported mail timezone: {timezone}") from exc


def _normalize_mail_datetime(value: datetime | None, timezone: ZoneInfo) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone).replace(tzinfo=None)


def _validate_join_url(join_url: str | None) -> None:
    if join_url is None:
        return
    parsed = urlparse(join_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("join_url must be an absolute http(s) URL")


def _status_from_metadata(metadata: dict | None) -> ApplicationStatus | None:
    raw = (metadata or {}).get("to_status")
    if not raw:
        return None
    try:
        return ApplicationStatus(str(raw))
    except ValueError:
        return None
    try:
        return ApplicationStatus(str(raw))
    except ValueError:
        return None


def _status_from_mail_event(event_type: str) -> ApplicationStatus:
    normalized = _normalize_text(event_type)
    if "offer" in normalized or "录用" in normalized or "入职" in normalized:
        return ApplicationStatus.OFFER
    if "reject" in normalized or "拒" in normalized:
        return ApplicationStatus.REJECTED
    if "ai" in normalized or "assessment" in normalized or "测评" in normalized:
        return ApplicationStatus.ASSESSMENT
    if "written" in normalized or "笔试" in normalized:
        return ApplicationStatus.WRITTEN_TEST
    if "interview" in normalized or "面试" in normalized:
        return ApplicationStatus.INTERVIEW_1
    return ApplicationStatus.APPLIED


def _mail_job_title(job_title: str | None) -> str:
    normalized = str(job_title or "").strip()
    return normalized or "邮件通知岗位（岗位未提供）"


def _mail_job_identity(company_name: str, job_title: str | None) -> str:
    from hashlib import sha256

    identity = f"{_normalize_text(company_name)}|{_normalize_text(job_title) if job_title else 'job-unspecified'}"
    return f"qq-mail:{sha256(identity.encode('utf-8')).hexdigest()}"


def _normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().split())


@dataclass(frozen=True)
class MailApplicationSyncResult:
    application: Application
    event: ApplicationEvent
    created_application: bool


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)
