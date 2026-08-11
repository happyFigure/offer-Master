import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class FakeLeadVerifier:
    def verify(self, lead):
        from app.domains.jobs.verification import LeadVerificationCheck

        return LeadVerificationCheck(
            is_open=True,
            verified_url=lead.apply_url,
            notes="official application URL is reachable",
        )


class LeadVerificationServiceTest(unittest.TestCase):
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

    def test_verified_open_lead_converts_to_formal_job(self):
        from app.domains.jobs.models import Job, JobSourceTrustLevel, JobSourceType
        from app.domains.jobs.repository import (
            CompanyRepository,
            JobLeadRepository,
            JobRepository,
            JobSourceRepository,
            RawJobLeadRepository,
            SourceSyncRunRepository,
        )
        from app.domains.jobs.schemas import JobLeadCreate, JobSourceCreate
        from app.domains.jobs.service import JobLeadService, JobService
        from app.domains.jobs.verification import LeadVerificationService

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
                    name="DLMU career site",
                    source_type=JobSourceType.UNIVERSITY_CAREER_SITE,
                    entry_url="https://career.example.edu/jobs",
                    trust_level=JobSourceTrustLevel.HIGH,
                    fetch_mode="public_html",
                )
            )
            lead = lead_service.create_lead(
                JobLeadCreate(
                    source_id=source.id,
                    company_name="Li Auto",
                    title="Backend Engineer",
                    city="Beijing",
                    source_url="https://career.example.edu/job/101",
                    apply_url="https://career.lixiang.com/campus/101",
                    job_type="campus",
                    skills=["Java"],
                )
            )

            result = LeadVerificationService(
                lead_service=lead_service,
                job_service=job_service,
                verifier=FakeLeadVerifier(),
            ).verify_and_convert(lead.id)
            session.commit()

            stored_job = session.scalars(select(Job)).one()

        self.assertTrue(result.created)
        self.assertEqual("converted", result.lead.verification_status)
        self.assertEqual("https://career.lixiang.com/campus/101", result.lead.verified_url)
        self.assertEqual("Li Auto", stored_job.company.name)
        self.assertEqual("job_lead", stored_job.source)


if __name__ == "__main__":
    unittest.main()
