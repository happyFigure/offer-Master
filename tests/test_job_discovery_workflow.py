import sys
import unittest
import warnings
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

    warnings.simplefilter("ignore", LangChainPendingDeprecationWarning)
except ImportError:
    pass


class FakeSocialProvider:
    def extract(self, source_id, raw_lead_id, raw_content, source_url, trust_level):
        from app.domains.jobs.schemas import JobLeadCreate

        return [
            JobLeadCreate(
                source_id=source_id,
                raw_lead_id=raw_lead_id,
                company_name="理想汽车",
                title="后端开发工程师",
                city="北京",
                job_direction="backend",
                graduation_year="2027",
                source_url=source_url,
                apply_url="https://www.lixiang.com/career",
                job_type="campus",
                jd_text="负责后端服务开发。",
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


class JobDiscoveryWorkflowTest(unittest.TestCase):
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

    def test_manual_social_lead_import_workflow_captures_raw_and_creates_leads(self):
        from app.agent_runtime.workflows.job_discovery import (
            ManualSocialLeadImportCommand,
            run_manual_social_lead_import,
        )
        from app.domains.jobs.models import JobLead, JobSourceType
        from app.domains.jobs.repository import (
            JobLeadRepository,
            JobSourceRepository,
            RawJobLeadRepository,
            SourceSyncRunRepository,
        )
        from app.domains.jobs.schemas import JobSourceCreate
        from app.domains.jobs.service import JobLeadService

        with self.Session() as session:
            service = JobLeadService(
                sources=JobSourceRepository(session),
                sync_runs=SourceSyncRunRepository(session),
                raw_leads=RawJobLeadRepository(session),
                leads=JobLeadRepository(session),
            )
            source = service.create_source(
                JobSourceCreate(
                    name="小红书-秋招汇总",
                    source_type=JobSourceType.XIAOHONGSHU_NOTE,
                    entry_url="https://www.xiaohongshu.com/discovery/item/demo",
                    trust_level="medium",
                    fetch_mode="manual_clip",
                )
            )
            result = run_manual_social_lead_import(
                ManualSocialLeadImportCommand(
                    source_id=source.id,
                    raw_content="理想汽车 2027 秋招 后端开发 北京 Java",
                    source_url="https://www.xiaohongshu.com/discovery/item/demo",
                ),
                lead_service=service,
                provider=FakeSocialProvider(),
            )
            session.commit()

            stored_leads = session.scalars(select(JobLead)).all()

        self.assertTrue(result.raw_capture.created)
        self.assertEqual(1, len(result.leads))
        self.assertEqual("理想汽车", stored_leads[0].company_name)
        self.assertEqual("unverified", stored_leads[0].verification_status)


    def test_university_career_sync_workflow_records_sync_run_and_creates_leads(self):
        from app.agent_runtime.workflows.job_discovery import (
            UniversityCareerSyncCommand,
            run_university_career_source_sync,
        )
        from app.domains.jobs.models import (
            JobLead,
            JobSourceFetchMode,
            JobSourceTrustLevel,
            JobSourceType,
            RawJobLead,
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

        with self.Session() as session:
            service = JobLeadService(
                sources=JobSourceRepository(session),
                sync_runs=SourceSyncRunRepository(session),
                raw_leads=RawJobLeadRepository(session),
                leads=JobLeadRepository(session),
            )
            source = service.create_source(
                JobSourceCreate(
                    name="DLMU career site",
                    source_type=JobSourceType.UNIVERSITY_CAREER_SITE,
                    entry_url="https://career.example.edu/jobs",
                    trust_level=JobSourceTrustLevel.HIGH,
                    fetch_mode=JobSourceFetchMode.PUBLIC_HTML,
                )
            )
            result = run_university_career_source_sync(
                UniversityCareerSyncCommand(source_id=source.id, limit=10),
                lead_service=service,
                content_provider=FakeUniversityProvider(),
                social_provider=FakeSocialProvider(),
            )
            session.commit()

            stored_runs = session.scalars(select(SourceSyncRun)).all()
            stored_raw = session.scalars(select(RawJobLead)).all()
            stored_leads = session.scalars(select(JobLead)).all()

        self.assertEqual("succeeded", result.sync_run.status)
        self.assertEqual(1, result.fetched_count)
        self.assertEqual(1, result.extracted_count)
        self.assertEqual(1, len(stored_runs))
        self.assertEqual("succeeded", stored_runs[0].status)
        self.assertEqual("extracted", stored_raw[0].status)
        self.assertEqual(stored_runs[0].id, stored_raw[0].sync_run_id)
        self.assertEqual(1, len(stored_leads))
        self.assertEqual("https://career.example.edu/job/101", stored_leads[0].source_url)

    def test_due_job_source_sync_workflow_processes_enabled_due_sources_only(self):
        from app.agent_runtime.workflows.job_discovery import (
            DueJobSourceSyncCommand,
            run_due_job_source_syncs,
        )
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
            due_source = service.create_source(
                JobSourceCreate(
                    name="Due university source",
                    source_type=JobSourceType.UNIVERSITY_CAREER_SITE,
                    entry_url="https://career.example.edu/jobs",
                    trust_level=JobSourceTrustLevel.HIGH,
                    fetch_mode=JobSourceFetchMode.PUBLIC_HTML,
                    sync_interval_hours=1,
                )
            )
            due_source.last_synced_at = now - timedelta(hours=2)
            never_synced_source = service.create_source(
                JobSourceCreate(
                    name="Never synced university source",
                    source_type=JobSourceType.UNIVERSITY_CAREER_SITE,
                    entry_url="https://career2.example.edu/jobs",
                    trust_level=JobSourceTrustLevel.HIGH,
                    fetch_mode=JobSourceFetchMode.PUBLIC_HTML,
                    sync_interval_hours=24,
                )
            )
            fresh_source = service.create_source(
                JobSourceCreate(
                    name="Fresh university source",
                    source_type=JobSourceType.UNIVERSITY_CAREER_SITE,
                    entry_url="https://fresh.example.edu/jobs",
                    trust_level=JobSourceTrustLevel.HIGH,
                    fetch_mode=JobSourceFetchMode.PUBLIC_HTML,
                    sync_interval_hours=24,
                )
            )
            fresh_source.last_synced_at = now - timedelta(hours=1)
            service.create_source(
                JobSourceCreate(
                    name="Disabled university source",
                    source_type=JobSourceType.UNIVERSITY_CAREER_SITE,
                    entry_url="https://disabled.example.edu/jobs",
                    trust_level=JobSourceTrustLevel.HIGH,
                    fetch_mode=JobSourceFetchMode.PUBLIC_HTML,
                    enabled=False,
                    sync_interval_hours=1,
                )
            )

            result = run_due_job_source_syncs(
                DueJobSourceSyncCommand(limit_per_source=5, now=now),
                lead_service=service,
                university_provider=FakeUniversityProvider(),
                social_provider=FakeSocialProvider(),
            )
            session.commit()

            stored_runs = session.scalars(select(SourceSyncRun)).all()
            stored_leads = session.scalars(select(JobLead)).all()

        self.assertEqual(2, result.succeeded_count)
        self.assertEqual(0, result.failed_count)
        self.assertEqual(1, result.skipped_count)
        self.assertEqual({due_source.id, never_synced_source.id}, set(result.processed_source_ids))
        self.assertEqual(2, len(stored_runs))
        self.assertEqual(2, len(stored_leads))

    def test_due_job_source_sync_graph_invokes_due_sync_workflow(self):
        from app.agent_runtime.workflows.job_discovery import (
            DueJobSourceSyncCommand,
            build_due_job_source_sync_graph,
        )
        from app.domains.jobs.models import (
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
            service.create_source(
                JobSourceCreate(
                    name="Graph due university source",
                    source_type=JobSourceType.UNIVERSITY_CAREER_SITE,
                    entry_url="https://career.example.edu/jobs",
                    trust_level=JobSourceTrustLevel.HIGH,
                    fetch_mode=JobSourceFetchMode.PUBLIC_HTML,
                )
            )
            graph = build_due_job_source_sync_graph(
                lead_service=service,
                university_provider=FakeUniversityProvider(),
                social_provider=FakeSocialProvider(),
            )
            state = graph.invoke({"command": DueJobSourceSyncCommand(limit_per_source=5, now=now)})
            session.commit()

            stored_runs = session.scalars(select(SourceSyncRun)).all()

        self.assertEqual(1, state["result"].succeeded_count)
        self.assertEqual(1, len(stored_runs))


if __name__ == "__main__":
    unittest.main()
