import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
