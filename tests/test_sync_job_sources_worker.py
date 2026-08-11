import importlib.util
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class FakeSocialProvider:
    def extract(self, source_id, raw_lead_id, raw_content, source_url, trust_level):
        from app.domains.jobs.schemas import JobLeadCreate

        return [
            JobLeadCreate(
                source_id=source_id,
                raw_lead_id=raw_lead_id,
                company_name="Li Auto",
                title="Java Backend Engineer",
                city="Beijing",
                job_direction="backend",
                graduation_year="2027",
                source_url=source_url,
                apply_url="https://career.lixiang.com/campus",
                job_type="campus",
                jd_text="Build Java backend services.",
                skills=["Java"],
                trust_level=trust_level,
            )
        ]


class FakeUniversityProvider:
    def fetch(self, entry_url, limit):
        from app.domains.jobs.providers.university_career import UniversityCareerEntry

        return [
            UniversityCareerEntry(
                title="Li Auto 2027 campus recruiting backend engineer",
                source_url="https://career.example.edu/job/101",
                raw_content="Li Auto 2027 campus recruiting backend engineer Beijing Java",
            )
        ]


def load_worker_module():
    worker_path = PROJECT_ROOT / "apps" / "worker" / "app" / "jobs" / "sync_job_sources.py"
    spec = importlib.util.spec_from_file_location("job_source_sync_worker", worker_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SyncJobSourcesWorkerTest(unittest.TestCase):
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

    def test_scheduled_sync_worker_runs_due_sources_once(self):
        from app.domains.jobs.models import (
            JobLead,
            JobSourceFetchMode,
            JobSourceTrustLevel,
            JobSourceType,
            SourceSyncRun,
        )
        from app.domains.jobs.repository import (
            JobLeadRepository,
            JobSourceRepository,
            RawJobLeadRepository,
            SourceSyncRunRepository,
        )
        from app.domains.jobs.schemas import JobSourceCreate
        from app.domains.jobs.service import JobLeadService

        now = datetime(2026, 8, 11, 9, 0, 0)
        with self.Session() as session:
            service = JobLeadService(
                sources=JobSourceRepository(session),
                sync_runs=SourceSyncRunRepository(session),
                raw_leads=RawJobLeadRepository(session),
                leads=JobLeadRepository(session),
            )
            source = service.create_source(
                JobSourceCreate(
                    name="Worker due university source",
                    source_type=JobSourceType.UNIVERSITY_CAREER_SITE,
                    entry_url="https://career.example.edu/jobs",
                    trust_level=JobSourceTrustLevel.HIGH,
                    fetch_mode=JobSourceFetchMode.PUBLIC_HTML,
                    sync_interval_hours=1,
                )
            )
            source.last_synced_at = now - timedelta(hours=3)
            session.commit()

        worker = load_worker_module()
        summary = worker.run_scheduled_sync_once(
            session_factory=self.Session,
            university_provider=FakeUniversityProvider(),
            social_provider=FakeSocialProvider(),
            now=now,
            limit_per_source=5,
        )

        with self.Session() as session:
            runs = session.scalars(select(SourceSyncRun)).all()
            leads = session.scalars(select(JobLead)).all()

        self.assertEqual(1, summary["processed"])
        self.assertEqual(1, summary["succeeded"])
        self.assertEqual(0, summary["failed"])
        self.assertEqual(1, len(runs))
        self.assertEqual(1, len(leads))


if __name__ == "__main__":
    unittest.main()
