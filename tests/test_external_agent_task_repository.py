import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class ExternalAgentTaskRepositoryTest(unittest.TestCase):
    def setUp(self):
        from app.agent_runtime.external_tasks import models as external_task_models  # noqa: F401
        import app.domains.automation.models  # noqa: F401
        from app.db.base import Base

        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self):
        self.engine.dispose()

    def test_service_persists_task_events_and_result_artifacts(self):
        from app.agent_runtime.external_tasks.models import (
            ExternalAgentArtifact,
            ExternalAgentTaskEvent,
        )
        from app.agent_runtime.external_tasks.repository import SqlAlchemyExternalAgentTaskRepository
        from app.agent_runtime.external_tasks.schemas import (
            ApplyEntryDiscoveryResult,
            ApplyEntryDiscoveryStatus,
            ExternalAgentArtifactRef,
            ExternalAgentTaskStatus,
            ExternalTaskCandidateProfileRef,
            ExternalTaskJobContext,
            FindApplyEntryTaskEnvelope,
        )
        from app.agent_runtime.external_tasks.service import ExternalAgentTaskService

        with self.Session() as session:
            repository = SqlAlchemyExternalAgentTaskRepository(session)
            service = ExternalAgentTaskService(repository)
            envelope = FindApplyEntryTaskEnvelope(
                task_id="task-sqlalchemy-1",
                trace_id="trace-sqlalchemy-1",
                objective="Find and open the official campus apply page; stop before submit.",
                job=ExternalTaskJobContext(
                    job_id="job-1",
                    company_name="Tencent",
                    title="Backend Engineer Intern",
                    source_url="https://careers.tencent.com/job/1",
                    apply_url_candidate="https://careers.tencent.com/apply/1",
                ),
                candidate_profile_ref=ExternalTaskCandidateProfileRef(
                    profile_id="profile-1",
                    resume_version_id="resume-2026-campus",
                ),
            )

            created = service.create_find_apply_entry_task(envelope)
            service.mark_running(created.task_id)
            service.record_result(
                created.task_id,
                ApplyEntryDiscoveryResult(
                    task_id=created.task_id,
                    status=ApplyEntryDiscoveryStatus.FOUND_OPENED,
                    confidence=0.91,
                    company_name="Tencent",
                    job_title="Backend Engineer Intern",
                    source_url="https://careers.tencent.com/job/1",
                    apply_url="https://careers.tencent.com/apply/1",
                    final_browser_url="https://careers.tencent.com/apply/1?from=campus",
                    platform="tencent_careers",
                    button_text="Apply Now",
                    evidence_artifacts=[
                        ExternalAgentArtifactRef(
                            artifact_type="screenshot",
                            path_or_uri="F:/pythonProject/OfferMaster/data/exports/task-sqlalchemy-1.png",
                            mime_type="image/png",
                            metadata={"button_selector": "button.apply"},
                        )
                    ],
                    notes="Opened the apply form and stopped before final submission.",
                ),
            )
            session.commit()

        with self.Session() as session:
            repository = SqlAlchemyExternalAgentTaskRepository(session)
            persisted = repository.get("task-sqlalchemy-1")
            events = list(
                session.scalars(
                    select(ExternalAgentTaskEvent)
                    .where(ExternalAgentTaskEvent.task_id == "task-sqlalchemy-1")
                    .order_by(ExternalAgentTaskEvent.created_at)
                ).all()
            )
            artifacts = list(
                session.scalars(
                    select(ExternalAgentArtifact)
                    .where(ExternalAgentArtifact.task_id == "task-sqlalchemy-1")
                    .order_by(ExternalAgentArtifact.created_at)
                ).all()
            )

        self.assertIsNotNone(persisted)
        assert persisted is not None
        self.assertEqual(ExternalAgentTaskStatus.SUCCEEDED, persisted.status)
        self.assertEqual("find_apply_entry", persisted.task_type)
        self.assertEqual("trace-sqlalchemy-1", persisted.input_payload["trace_id"])
        self.assertEqual("https://careers.tencent.com/apply/1", persisted.output_payload["apply_url"])
        self.assertEqual(["task_queued", "task_running", "task_succeeded"], [e.event_type for e in events])
        self.assertEqual(1, len(artifacts))
        self.assertEqual("screenshot", artifacts[0].artifact_type)
        self.assertEqual("image/png", artifacts[0].mime_type)
        self.assertEqual({"button_selector": "button.apply"}, artifacts[0].artifact_metadata)


if __name__ == "__main__":
    unittest.main()
