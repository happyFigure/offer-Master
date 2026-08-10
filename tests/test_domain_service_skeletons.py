import sys
import unittest

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class DomainServiceSkeletonsTest(unittest.TestCase):
    def setUp(self):
        from app.db.base import Base
        from app.domains.applications import models as application_models  # noqa: F401
        from app.domains.automation import models as automation_models  # noqa: F401
        from app.domains.jobs import models as job_models  # noqa: F401

        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self):
        self.engine.dispose()

    def test_job_service_imports_job_once_by_source_identity(self):
        from app.domains.jobs.models import Company, Job
        from app.domains.jobs.repository import CompanyRepository, JobRepository
        from app.domains.jobs.schemas import JobImportDraft
        from app.domains.jobs.service import JobService

        with self.Session() as session:
            service = JobService(
                companies=CompanyRepository(session),
                jobs=JobRepository(session),
            )
            draft = JobImportDraft(
                company_name="Acme AI",
                title="AI Application Engineer",
                source="mock",
                source_job_id="mock-001",
                source_url="https://example.com/jobs/mock-001",
                skills=["Python", "LLM"],
            )

            first = service.import_job(draft)
            second = service.import_job(draft)
            third = service.import_job(draft.model_copy(update={"company_name": "Acme AI China"}))
            session.commit()

            stored_companies = session.scalars(select(Company)).all()
            stored_jobs = session.scalars(select(Job)).all()

        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertFalse(third.created)
        self.assertEqual(first.job.id, second.job.id)
        self.assertEqual(first.job.id, third.job.id)
        self.assertEqual("acme ai", first.company.normalized_name)
        self.assertEqual(["Python", "LLM"], first.job.skills)
        self.assertEqual("JobImported", first.event.event_type)
        self.assertEqual(1, len(stored_companies))
        self.assertEqual(1, len(stored_jobs))

    def test_application_service_creates_application_with_timeline_event(self):
        from app.domains.applications.models import ApplicationEvent, ApplicationStatus
        from app.domains.applications.repository import (
            ApplicationEventRepository,
            ApplicationRepository,
        )
        from app.domains.applications.schemas import ApplicationCreate
        from app.domains.applications.service import ApplicationService
        from app.domains.jobs.models import Company, Job, JobStatus

        with self.Session() as session:
            company = Company(name="Timeline Inc", normalized_name="timeline inc")
            job = Job(
                company=company,
                title="Backend Engineer",
                source="mock",
                source_job_id="timeline-001",
                skills=[],
                status=JobStatus.OPEN,
            )
            session.add(job)
            session.flush()

            service = ApplicationService(
                applications=ApplicationRepository(session),
                events=ApplicationEventRepository(session),
            )
            result = service.create_application(
                ApplicationCreate(
                    job_id=job.id,
                    status=ApplicationStatus.PREPARING,
                    priority="high",
                    channel="manual",
                    notes="Prepare tailored resume.",
                )
            )
            session.commit()

            stored_events = session.scalars(select(ApplicationEvent)).all()

        self.assertEqual(ApplicationStatus.PREPARING, result.application.status)
        self.assertEqual("application_created", stored_events[0].event_type)
        self.assertEqual(ApplicationStatus.PREPARING, stored_events[0].to_status)
        self.assertEqual("ApplicationCreated", result.event.event_type)

    def test_automation_service_requests_user_approval_boundary(self):
        from app.domains.automation.models import ApprovalRequestStatus, WorkflowRunStatus
        from app.domains.automation.repository import (
            ApprovalRequestRepository,
            ToolCallLogRepository,
            WorkflowCheckpointRepository,
            WorkflowRunRepository,
        )
        from app.domains.automation.schemas import ApprovalRequestCreate, WorkflowRunCreate
        from app.domains.automation.service import AutomationService

        with self.Session() as session:
            service = AutomationService(
                workflow_runs=WorkflowRunRepository(session),
                checkpoints=WorkflowCheckpointRepository(session),
                tool_call_logs=ToolCallLogRepository(session),
                approvals=ApprovalRequestRepository(session),
            )

            workflow = service.start_workflow(
                WorkflowRunCreate(
                    workflow_type="application_assist",
                    current_step="prepare_materials",
                    user_goal="Submit a real application after confirmation.",
                )
            )
            approval_result = service.request_user_approval(
                ApprovalRequestCreate(
                    workflow_run_id=workflow.id,
                    action_type="submit_application",
                    prompt="Confirm before real submission.",
                    payload={"risk_level": "L3"},
                )
            )
            session.commit()

        self.assertEqual(WorkflowRunStatus.WAITING_USER, workflow.status)
        self.assertEqual(approval_result.approval.id, workflow.approval_request_id)
        self.assertEqual(ApprovalRequestStatus.PENDING, approval_result.approval.status)
        self.assertEqual("AutomationWaitingForUser", approval_result.event.event_type)

    def test_deferred_domains_have_explicit_skeletons(self):
        from app.domains.interviews.events import InterviewPracticeQueued
        from app.domains.interviews.repository import InterviewRepository
        from app.domains.interviews.schemas import InterviewPracticeDraft
        from app.domains.interviews.service import InterviewService
        from app.domains.knowledge.events import KnowledgeDocumentIngestionQueued
        from app.domains.knowledge.repository import KnowledgeRepository
        from app.domains.knowledge.schemas import KnowledgeDocumentDraft
        from app.domains.knowledge.service import KnowledgeService

        knowledge_draft = KnowledgeDocumentDraft(
            title="Resume",
            source_path="F:/pythonProject/OfferMaster/data/imports/resume.md",
        )
        interview_draft = InterviewPracticeDraft(application_id="app-001", question="Why us?")

        self.assertEqual("interview_phase", InterviewService.deferred_until)
        self.assertEqual("rag_phase", KnowledgeService.deferred_until)
        self.assertEqual("Resume", knowledge_draft.title)
        self.assertEqual("Why us?", interview_draft.question)
        self.assertEqual(
            "KnowledgeDocumentIngestionQueued",
            KnowledgeDocumentIngestionQueued(document_id="doc-001").event_type,
        )
        self.assertEqual(
            "InterviewPracticeQueued",
            InterviewPracticeQueued(application_id="app-001").event_type,
        )
        with self.assertRaises(NotImplementedError):
            KnowledgeRepository().get("doc-001")
        with self.assertRaises(NotImplementedError):
            InterviewRepository().get_session("session-001")


if __name__ == "__main__":
    unittest.main()
