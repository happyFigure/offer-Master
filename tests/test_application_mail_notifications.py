import sys
import unittest
from asyncio import run
from datetime import datetime
from pathlib import Path

from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class ApplicationMailNotificationTests(unittest.TestCase):
    def setUp(self):
        from app.db.base import Base
        from app.domains.agent_memory import models as agent_memory_models  # noqa: F401
        from app.domains.applications import models as application_models  # noqa: F401
        from app.domains.automation import models as automation_models  # noqa: F401
        from app.domains.jobs import models as job_models  # noqa: F401

        self.engine = create_engine(
            "sqlite+pysqlite://",
            connect_args={"check_same_thread": False},
            future=True,
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self):
        self.engine.dispose()

    def _create_application(self):
        from app.domains.applications.models import Application, ApplicationStatus
        from app.domains.jobs.models import Company, Job, JobStatus

        with self.Session() as session:
            company = Company(name="Acme AI", normalized_name="acme ai")
            job = Job(
                company=company,
                title="后端开发工程师",
                city="上海",
                source="manual",
                source_job_id="acme-backend-001",
                source_url="https://example.com/jobs/acme-backend-001",
                job_type="秋招",
                status=JobStatus.OPEN,
            )
            application = Application(job=job, status=ApplicationStatus.APPLIED, priority="high")
            session.add(application)
            session.commit()
            return application.id

    def test_mail_notification_is_pending_and_idempotent_before_confirmation(self):
        from app.domains.applications.models import ApplicationEvent
        from app.domains.applications.schemas import ApplicationMailNotificationCreate
        from app.domains.applications.service import ApplicationService
        from app.domains.applications.repository import ApplicationEventRepository, ApplicationRepository

        application_id = self._create_application()
        command = ApplicationMailNotificationCreate(
            application_id=application_id,
            event_type="written_test",
            title="在线笔试通知",
            company_name="Acme AI",
            job_title="后端开发工程师",
            deadline_at=datetime(2026, 10, 5, 23, 59),
            timezone="Asia/Shanghai",
            confidence=0.42,
            source_message_id="<mail-001@example.com>",
            source_uid="42",
            evidence=["请于10月5日23:59前完成笔试"],
        )

        with self.Session() as session:
            service = ApplicationService(ApplicationRepository(session), ApplicationEventRepository(session))
            first = service.create_mail_notification_candidate(command)
            duplicate = service.create_mail_notification_candidate(command)
            session.commit()

            events = session.scalars(select(ApplicationEvent)).all()
            application = service._applications.get(application_id)

        self.assertIs(first, duplicate)
        self.assertEqual(1, len(events))
        self.assertEqual("pending", first.review_status)
        self.assertEqual("<mail-001@example.com>", first.source_message_id)
        self.assertEqual(datetime(2026, 10, 5, 23, 59), first.deadline_at)
        self.assertEqual("Asia/Shanghai", first.timezone)
        self.assertEqual("applied", str(application.status))

    def test_confirming_notification_updates_application_once_and_keeps_mail_evidence(self):
        from app.domains.applications.models import ApplicationEvent
        from app.domains.applications.repository import ApplicationEventRepository, ApplicationRepository
        from app.domains.applications.schemas import ApplicationMailNotificationCreate
        from app.domains.applications.service import ApplicationService

        application_id = self._create_application()
        with self.Session() as session:
            service = ApplicationService(ApplicationRepository(session), ApplicationEventRepository(session))
            candidate = service.create_mail_notification_candidate(
                ApplicationMailNotificationCreate(
                    application_id=application_id,
                    event_type="interview",
                    title="一面邀请",
                    company_name="Acme AI",
                    job_title="后端开发工程师",
                    scheduled_at=datetime(2026, 10, 8, 14, 0),
                    timezone="Asia/Shanghai",
                    join_url="https://example.com/interview/1",
                    confidence=0.96,
                    source_message_id="<mail-002@example.com>",
                    evidence=["面试时间：10月8日 14:00"],
                    to_status="interview_1",
                )
            )
            application = service.confirm_mail_notification_candidate(candidate.id)
            same_application = service.confirm_mail_notification_candidate(candidate.id)
            session.commit()

            events = session.scalars(select(ApplicationEvent).where(ApplicationEvent.application_id == application_id)).all()

        self.assertEqual("interview_1", getattr(application.status, "value", application.status))
        self.assertEqual(application.id, same_application.id)
        mail_event = next(event for event in events if event.source == "qq_mail")
        self.assertEqual("confirmed", mail_event.review_status)
        self.assertEqual("<mail-002@example.com>", mail_event.source_message_id)
        self.assertEqual("https://example.com/interview/1", mail_event.join_url)
        self.assertTrue(any(event.event_type == "status_changed" for event in events))
        self.assertEqual(1, sum(event.event_type == "status_changed" for event in events))

    def test_notification_api_exposes_pending_review_and_confirm_action(self):
        from app.db.session import get_db_session
        from app.main import create_app

        application_id = self._create_application()
        app = create_app()

        def override_session():
            with self.Session() as session:
                yield session

        app.dependency_overrides[get_db_session] = override_session

        async def request_flow():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                created = await client.post(
                    f"/api/v1/applications/{application_id}/notification-candidates",
                    json={
                        "application_id": application_id,
                        "event_type": "assessment",
                        "title": "测评通知",
                        "company_name": "Acme AI",
                        "confidence": 0.88,
                        "source_message_id": "<mail-003@example.com>",
                        "evidence": ["完成测评"],
                    },
                )
                pending = await client.get("/api/v1/applications/notification-candidates")
                event_id = pending.json()["items"][0]["id"]
                confirmed = await client.post(
                    f"/api/v1/applications/notification-candidates/{event_id}/confirm",
                    json={"to_status": "assessment"},
                )
                return created, pending, confirmed

        created, pending, confirmed = run(request_flow())
        self.assertEqual(200, created.status_code)
        self.assertEqual(200, pending.status_code)
        self.assertEqual("pending", pending.json()["items"][0]["review_status"])
        self.assertEqual(200, confirmed.status_code)
        self.assertEqual("confirmed", confirmed.json()["review_status"])

    def test_mail_sync_is_primary_tool_and_pending_only_legacy_tool_is_not_registered(self):
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry

        registry = create_default_agent_tool_registry()
        definition = registry.get("applications.sync_mail_application")

        self.assertIsNotNone(definition)
        self.assertIsNone(registry.get("applications.create_mail_notification_candidate"))
        self.assertFalse(definition.requires_confirmation)
        self.assertEqual("medium", definition.risk_level.value)
        self.assertIn("company_name", definition.input_schema["required"])
        self.assertNotIn("job_title", definition.input_schema["required"])

    def test_mail_matching_tool_reads_existing_applications_without_writing(self):
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry

        application_id = self._create_application()
        with self.Session() as session:
            definition = create_default_agent_tool_registry().get("applications.find_for_mail_notification")
            result = definition.handler(session, company_name="Acme AI", job_title="后端")
            session.rollback()

        self.assertTrue(result["ok"])
        self.assertEqual(application_id, result["result"]["matches"][0]["application_id"])
        self.assertEqual("applied", result["result"]["matches"][0]["status"])

    def test_sync_unmatched_mail_creates_application_and_preserves_mail_evidence(self):
        from app.domains.applications.models import Application, ApplicationEvent, ApplicationStatus
        from app.domains.applications.repository import ApplicationEventRepository, ApplicationRepository
        from app.domains.applications.schemas import ApplicationMailSyncCreate
        from app.domains.applications.service import ApplicationService
        from app.domains.jobs.repository import CompanyRepository, JobRepository
        from app.domains.jobs.service import JobService

        command = ApplicationMailSyncCreate(
            company_name="欣旺达电子股份有限公司",
            event_type="ai_interview",
            title="请尽快完成欣旺达电子股份有限公司发起的AI面试",
            body="请在 2026-10-02 23:59 前完成 AI 面试",
            deadline_at=datetime(2026, 10, 2, 23, 59),
            timezone="Asia/Shanghai",
            join_url="https://example.com/sunwoda-ai-interview",
            confidence=0.98,
            source_message_id="<sunwoda-ai-001@example.com>",
            source_uid="1577",
            evidence=["你有一场AI面试即将到期", "2026-10-02 23:59"],
        )

        with self.Session() as session:
            service = ApplicationService(
                ApplicationRepository(session),
                ApplicationEventRepository(session),
                jobs=JobService(CompanyRepository(session), JobRepository(session)),
            )
            first = service.sync_mail_application(command)
            duplicate = service.sync_mail_application(command)
            session.commit()

            applications = session.scalars(select(Application)).all()
            events = session.scalars(
                select(ApplicationEvent).where(ApplicationEvent.source == "qq_mail")
            ).all()

        self.assertEqual(first.application.id, duplicate.application.id)
        self.assertEqual(1, len(applications))
        self.assertEqual("欣旺达电子股份有限公司", first.application.job.company.name)
        self.assertEqual("邮件通知岗位（岗位未提供）", first.application.job.title)
        self.assertEqual(ApplicationStatus.ASSESSMENT, first.application.status)
        self.assertEqual(1, len(events))
        self.assertEqual(first.application.id, events[0].application_id)
        self.assertEqual("<sunwoda-ai-001@example.com>", events[0].source_message_id)
        self.assertEqual("synced", events[0].review_status)

    def test_sync_mail_requires_company_but_allows_missing_job_title(self):
        from pydantic import ValidationError
        from app.domains.applications.schemas import ApplicationMailSyncCreate

        with self.assertRaises(ValidationError):
            ApplicationMailSyncCreate(
                event_type="ai_interview",
                title="AI 面试通知",
                confidence=0.9,
                source_message_id="<missing-company@example.com>",
            )

    def test_mail_sync_tool_creates_board_record_from_structured_agent_result(self):
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry
        from app.domains.applications.models import Application

        definition = create_default_agent_tool_registry().get("applications.sync_mail_application")
        self.assertIsNotNone(definition)
        self.assertIn("company_name", definition.input_schema["required"])
        self.assertNotIn("job_title", definition.input_schema["required"])

        with self.Session() as session:
            result = definition.handler(
                session,
                company_name="欣旺达电子股份有限公司",
                event_type="ai_interview",
                title="欣旺达 AI 面试通知",
                confidence=0.98,
                source_message_id="<sunwoda-tool-001@example.com>",
                source_uid="1577",
                deadline_at="2026-10-02T23:59:00",
                timezone="Asia/Shanghai",
                evidence=["请在截止时间前完成 AI 面试"],
            )
            application = session.scalars(select(Application)).one()

        self.assertTrue(result["ok"])
        self.assertEqual("assessment", result["result"]["status"])
        self.assertEqual("欣旺达电子股份有限公司", application.job.company.name)
        self.assertEqual("邮件通知岗位（岗位未提供）", application.job.title)

    def test_sync_mail_preserves_explicit_interview_start_and_deadline(self):
        from app.domains.applications.repository import ApplicationEventRepository, ApplicationRepository
        from app.domains.applications.schemas import ApplicationMailSyncCreate
        from app.domains.applications.service import ApplicationService
        from app.domains.jobs.repository import CompanyRepository, JobRepository
        from app.domains.jobs.service import JobService

        command = ApplicationMailSyncCreate(
            company_name="去哪儿旅行",
            event_type="ai_interview",
            title="在线 AI 面试通知",
            source_sent_at=datetime(2026, 10, 1, 9, 0),
            scheduled_at=datetime(2026, 10, 2, 14, 0),
            deadline_at=datetime(2026, 10, 6, 22, 0),
            timezone="Asia/Shanghai",
            confidence=0.98,
            source_message_id="<explicit-interview-time@example.com>",
        )

        with self.Session() as session:
            result = ApplicationService(
                ApplicationRepository(session),
                ApplicationEventRepository(session),
                jobs=JobService(CompanyRepository(session), JobRepository(session)),
            ).sync_mail_application(command)
            session.commit()

        self.assertEqual(datetime(2026, 10, 2, 14, 0), result.event.scheduled_at)
        self.assertEqual(datetime(2026, 10, 6, 22, 0), result.event.deadline_at)
        self.assertEqual("explicit", result.event.event_metadata["timing_source"])
        self.assertEqual("2026-10-01T09:00:00", result.event.event_metadata["source_sent_at"])

    def test_sync_mail_calculates_deadline_from_sent_time_and_relative_window(self):
        from app.domains.applications.repository import ApplicationEventRepository, ApplicationRepository
        from app.domains.applications.schemas import ApplicationMailSyncCreate
        from app.domains.applications.service import ApplicationService
        from app.domains.jobs.repository import CompanyRepository, JobRepository
        from app.domains.jobs.service import JobService

        command = ApplicationMailSyncCreate(
            company_name="浩鲸云计算科技股份有限公司",
            event_type="online_assessment",
            title="请在 48 小时内完成在线测评",
            source_sent_at=datetime(2026, 10, 1, 17, 10),
            deadline_offset_hours=48,
            timezone="Asia/Shanghai",
            confidence=0.95,
            source_message_id="<relative-deadline@example.com>",
        )

        with self.Session() as session:
            result = ApplicationService(
                ApplicationRepository(session),
                ApplicationEventRepository(session),
                jobs=JobService(CompanyRepository(session), JobRepository(session)),
            ).sync_mail_application(command)
            session.commit()

        self.assertIsNone(result.event.scheduled_at)
        self.assertEqual(datetime(2026, 10, 3, 17, 10), result.event.deadline_at)
        self.assertEqual("relative", result.event.event_metadata["timing_source"])
        self.assertEqual(48, result.event.event_metadata["deadline_offset_hours"])

    def test_explicit_deadline_wins_over_relative_window(self):
        from app.domains.applications.repository import ApplicationEventRepository, ApplicationRepository
        from app.domains.applications.schemas import ApplicationMailSyncCreate
        from app.domains.applications.service import ApplicationService
        from app.domains.jobs.repository import CompanyRepository, JobRepository
        from app.domains.jobs.service import JobService

        command = ApplicationMailSyncCreate(
            company_name="欣旺达电子股份有限公司",
            event_type="ai_interview",
            title="AI 面试通知",
            source_sent_at=datetime(2026, 10, 1, 9, 0),
            deadline_at=datetime(2026, 10, 2, 23, 59),
            deadline_offset_hours=48,
            confidence=0.98,
            source_message_id="<explicit-deadline-wins@example.com>",
        )

        with self.Session() as session:
            result = ApplicationService(
                ApplicationRepository(session),
                ApplicationEventRepository(session),
                jobs=JobService(CompanyRepository(session), JobRepository(session)),
            ).sync_mail_application(command)

        self.assertEqual(datetime(2026, 10, 2, 23, 59), result.event.deadline_at)
        self.assertEqual("explicit", result.event.event_metadata["timing_source"])

    def test_relative_deadline_requires_source_sent_at(self):
        from pydantic import ValidationError
        from app.domains.applications.schemas import ApplicationMailSyncCreate

        with self.assertRaises(ValidationError):
            ApplicationMailSyncCreate(
                company_name="Acme AI",
                event_type="assessment",
                title="请在 24 小时内完成测评",
                deadline_offset_hours=24,
                confidence=0.9,
                source_message_id="<missing-sent-at@example.com>",
            )

    def test_application_board_item_exposes_latest_mail_timing(self):
        from app.api.v1.applications import _application_board_item
        from app.domains.applications.repository import ApplicationEventRepository, ApplicationRepository
        from app.domains.applications.schemas import ApplicationMailSyncCreate
        from app.domains.applications.service import ApplicationService
        from app.domains.jobs.repository import CompanyRepository, JobRepository
        from app.domains.jobs.service import JobService

        with self.Session() as session:
            result = ApplicationService(
                ApplicationRepository(session),
                ApplicationEventRepository(session),
                jobs=JobService(CompanyRepository(session), JobRepository(session)),
            ).sync_mail_application(
                ApplicationMailSyncCreate(
                    company_name="去哪儿旅行",
                    event_type="interview",
                    title="面试安排",
                    source_sent_at=datetime(2026, 10, 1, 9, 0),
                    scheduled_at=datetime(2026, 10, 2, 14, 0),
                    deadline_at=datetime(2026, 10, 2, 18, 0),
                    timezone="Asia/Shanghai",
                    confidence=0.98,
                    source_message_id="<board-timing@example.com>",
                )
            )
            session.commit()
            application_id = result.application.id

        with self.Session() as session:
            application = ApplicationRepository(session).get(application_id)
            board_item = _application_board_item(application)

        self.assertIsNotNone(board_item.mail_timing)
        self.assertEqual(datetime(2026, 10, 2, 14, 0), board_item.mail_timing.scheduled_at)
        self.assertEqual(datetime(2026, 10, 2, 18, 0), board_item.mail_timing.deadline_at)
        self.assertEqual("explicit", board_item.mail_timing.timing_source)

    def test_resync_enriches_existing_mail_event_timing_without_duplicate(self):
        from app.domains.applications.models import ApplicationEvent
        from app.domains.applications.repository import ApplicationEventRepository, ApplicationRepository
        from app.domains.applications.schemas import ApplicationMailSyncCreate
        from app.domains.applications.service import ApplicationService
        from app.domains.jobs.repository import CompanyRepository, JobRepository
        from app.domains.jobs.service import JobService

        base = dict(
            company_name="某公司",
            event_type="online_assessment",
            title="在线测评通知",
            confidence=0.9,
            source_message_id="<enrich-mail-timing@example.com>",
        )
        with self.Session() as session:
            service = ApplicationService(
                ApplicationRepository(session),
                ApplicationEventRepository(session),
                jobs=JobService(CompanyRepository(session), JobRepository(session)),
            )
            first = service.sync_mail_application(ApplicationMailSyncCreate(**base))
            second = service.sync_mail_application(
                ApplicationMailSyncCreate(
                    **base,
                    source_sent_at=datetime(2026, 10, 1, 12, 0),
                    deadline_offset_hours=24,
                )
            )
            session.commit()
            event_count = session.query(ApplicationEvent).filter(ApplicationEvent.source == "qq_mail").count()

        self.assertEqual(first.event.id, second.event.id)
        self.assertEqual(datetime(2026, 10, 2, 12, 0), second.event.deadline_at)
        self.assertEqual("relative", second.event.event_metadata["timing_source"])
        self.assertEqual(1, event_count)


if __name__ == "__main__":
    unittest.main()
