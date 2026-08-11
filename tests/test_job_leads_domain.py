import sys
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class JobLeadDomainTest(unittest.TestCase):
    def setUp(self):
        from app.db.base import Base
        from app.domains.applications import models as application_models  # noqa: F401
        from app.domains.automation import models as automation_models  # noqa: F401
        from app.domains.jobs import models as job_models  # noqa: F401
        from sqlalchemy import create_engine

        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self):
        self.engine.dispose()

    def test_source_service_captures_raw_content_once_by_hash(self):
        from app.domains.jobs.models import JobSourceType, RawJobLead
        from app.domains.jobs.repository import (
            JobLeadRepository,
            JobSourceRepository,
            RawJobLeadRepository,
            SourceSyncRunRepository,
        )
        from app.domains.jobs.schemas import JobSourceCreate, RawJobLeadCreate
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

            first = service.capture_raw_lead(
                RawJobLeadCreate(
                    source_id=source.id,
                    source_url="https://www.xiaohongshu.com/discovery/item/demo",
                    raw_content="小米 2027 秋招 后端开发 北京 Java",
                    content_type="text/plain",
                )
            )
            second = service.capture_raw_lead(
                RawJobLeadCreate(
                    source_id=source.id,
                    source_url="https://www.xiaohongshu.com/discovery/item/demo",
                    raw_content="小米 2027 秋招 后端开发 北京 Java",
                    content_type="text/plain",
                )
            )
            session.commit()

            stored_raw_leads = session.scalars(select(RawJobLead)).all()

        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertEqual(first.raw_lead.id, second.raw_lead.id)
        self.assertEqual(1, len(stored_raw_leads))
        self.assertEqual(JobSourceType.XIAOHONGSHU_NOTE, source.source_type)
        self.assertEqual("JobLeadCaptured", first.event.event_type)

    def test_verified_lead_converts_to_formal_job(self):
        from app.domains.jobs.models import Job, JobLeadStatus, JobSourceType
        from app.domains.jobs.repository import (
            CompanyRepository,
            JobLeadRepository,
            JobRepository,
            JobSourceRepository,
            RawJobLeadRepository,
            SourceSyncRunRepository,
        )
        from app.domains.jobs.schemas import (
            JobLeadCreate,
            JobLeadVerification,
            JobSourceCreate,
            RawJobLeadCreate,
        )
        from app.domains.jobs.service import JobLeadService, JobService

        with self.Session() as session:
            lead_service = JobLeadService(
                sources=JobSourceRepository(session),
                sync_runs=SourceSyncRunRepository(session),
                raw_leads=RawJobLeadRepository(session),
                leads=JobLeadRepository(session),
            )
            job_service = JobService(
                companies=CompanyRepository(session),
                jobs=JobRepository(session),
            )
            source = lead_service.create_source(
                JobSourceCreate(
                    name="大连海事就业公众号",
                    source_type=JobSourceType.WECHAT_ARTICLE,
                    entry_url="https://mp.weixin.qq.com/s/example",
                    trust_level="medium_high",
                    fetch_mode="manual_clip",
                )
            )
            raw_result = lead_service.capture_raw_lead(
                RawJobLeadCreate(
                    source_id=source.id,
                    source_url="https://mp.weixin.qq.com/s/example",
                    raw_content="招银网络科技2027秋招 后端开发 深圳 Java Spring",
                    content_type="text/plain",
                )
            )
            lead = lead_service.create_lead(
                JobLeadCreate(
                    source_id=source.id,
                    raw_lead_id=raw_result.raw_lead.id,
                    company_name="招银网络科技",
                    title="后端开发工程师",
                    city="深圳",
                    job_direction="backend",
                    graduation_year="2027",
                    source_url="https://mp.weixin.qq.com/s/example",
                    apply_url="https://cmbntjob.cmbchina.com",
                    job_type="campus",
                    jd_text="负责 Java 后端服务开发。",
                    skills=["Java", "Spring"],
                    confidence_score=86.5,
                )
            )

            verified = lead_service.verify_lead(
                lead.id,
                JobLeadVerification(
                    verification_status=JobLeadStatus.VERIFIED,
                    verified_url="https://cmbntjob.cmbchina.com",
                    verification_notes="官网入口可访问。",
                ),
            )
            verified_status_before_conversion = verified.verification_status
            converted = lead_service.convert_verified_lead_to_job(lead.id, job_service)
            session.commit()

            stored_jobs = session.scalars(select(Job)).all()

        self.assertEqual(JobLeadStatus.VERIFIED, verified_status_before_conversion)
        self.assertEqual(JobLeadStatus.CONVERTED, converted.lead.verification_status)
        self.assertEqual(converted.job.id, converted.lead.converted_job_id)
        self.assertEqual("招银网络科技", converted.job.company.name)
        self.assertEqual("后端开发工程师", converted.job.title)
        self.assertEqual("job_lead", converted.job.source)
        self.assertEqual(1, len(stored_jobs))
        self.assertEqual("JobLeadConverted", converted.event.event_type)

    def test_third_migration_creates_source_and_lead_tables(self):
        migration = (
            PROJECT_ROOT
            / "infra"
            / "migrations"
            / "versions"
            / "20260811_0003_job_source_and_lead_tables.py"
        )
        self.assertTrue(migration.is_file())

        from app.core.config import get_settings

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as tmp_dir:
            db_path = Path(tmp_dir) / "job_leads_migration_check.sqlite"
            config = Config(str(PROJECT_ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(PROJECT_ROOT / "infra" / "migrations"))

            with patch.dict(
                "os.environ",
                {"JOBPILOT_DATABASE_URL": f"sqlite+pysqlite:///{db_path.as_posix()}"},
                clear=False,
            ):
                get_settings.cache_clear()
                command.upgrade(config, "head")

            engine = create_engine(f"sqlite+pysqlite:///{db_path.as_posix()}", future=True)
            inspector = inspect(engine)

            self.assertTrue(
                {
                    "job_sources",
                    "source_sync_runs",
                    "raw_job_leads",
                    "job_leads",
                }.issubset(set(inspector.get_table_names()))
            )
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
