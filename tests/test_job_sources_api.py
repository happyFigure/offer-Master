import sys
import unittest
from asyncio import run
from pathlib import Path

from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class FakeSocialLeadProvider:
    def extract(self, source_id, raw_lead_id, raw_content, source_url, trust_level):
        from app.domains.jobs.schemas import JobLeadCreate

        return [
            JobLeadCreate(
                source_id=source_id,
                raw_lead_id=raw_lead_id,
                company_name="Xiaomi",
                title="Java Backend Engineer",
                city="Beijing",
                job_direction="backend",
                graduation_year="2027",
                source_url=source_url,
                apply_url="https://hr.xiaomi.com/campus",
                job_type="campus",
                jd_text="Build Java backend services.",
                skills=["Java", "Spring"],
                confidence_score=89.0,
                trust_level=trust_level,
            )
        ]


class FakeUniversityCareerProvider:
    def fetch(self, entry_url, limit):
        from app.domains.jobs.providers.university_career import UniversityCareerEntry

        return [
            UniversityCareerEntry(
                title="Li Auto 2027 campus recruiting backend engineer",
                source_url="https://career.example.edu/job/101",
                raw_content="Li Auto 2027 campus recruiting backend engineer Beijing Java",
            )
        ]


class FakeOpenLeadVerifier:
    def verify(self, lead):
        from app.domains.jobs.verification import LeadVerificationCheck

        return LeadVerificationCheck(
            is_open=True,
            verified_url=lead.apply_url,
            notes="official application URL is reachable",
        )


class JobSourcesApiTest(unittest.TestCase):
    def setUp(self):
        from app.db.base import Base
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

    def _app(self, social_provider=None, university_provider=None, lead_verifier=None):
        from app.api.v1.job_sources import (
            get_lead_verifier,
            get_social_lead_provider,
            get_university_career_provider,
        )
        from app.db.session import get_db_session
        from app.main import create_app

        app = create_app()

        def override_session():
            with self.Session() as session:
                yield session

        app.dependency_overrides[get_db_session] = override_session
        if social_provider is not None:
            app.dependency_overrides[get_social_lead_provider] = lambda: social_provider
        if university_provider is not None:
            app.dependency_overrides[get_university_career_provider] = lambda: university_provider
        if lead_verifier is not None:
            app.dependency_overrides[get_lead_verifier] = lambda: lead_verifier
        return app

    def test_create_and_list_job_sources(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                created = await client.post(
                    "/api/v1/job-sources",
                    json={
                        "name": "小红书-薯条派秋招汇总",
                        "source_type": "xiaohongshu_note",
                        "entry_url": "https://www.xiaohongshu.com/discovery/item/demo",
                        "trust_level": "medium",
                        "fetch_mode": "manual_clip",
                    },
                )
                listed = await client.get("/api/v1/job-sources")
                return created, listed

        created_response, list_response = run(call_api())

        self.assertEqual(201, created_response.status_code)
        self.assertEqual(200, list_response.status_code)
        self.assertEqual("小红书-薯条派秋招汇总", created_response.json()["name"])
        self.assertEqual("xiaohongshu_note", list_response.json()["items"][0]["source_type"])

    def test_create_verify_and_convert_job_lead(self):
        from app.domains.jobs.models import Job

        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                source = await client.post(
                    "/api/v1/job-sources",
                    json={
                        "name": "大连海事就业公众号",
                        "source_type": "wechat_article",
                        "entry_url": "https://mp.weixin.qq.com/s/example",
                        "trust_level": "medium_high",
                        "fetch_mode": "manual_clip",
                    },
                )
                source_id = source.json()["id"]
                raw = await client.post(
                    "/api/v1/job-leads/raw",
                    json={
                        "source_id": source_id,
                        "source_url": "https://mp.weixin.qq.com/s/example",
                        "raw_content": "招银网络科技2027秋招 后端开发 深圳 Java Spring",
                        "content_type": "text/plain",
                    },
                )
                lead = await client.post(
                    "/api/v1/job-leads",
                    json={
                        "source_id": source_id,
                        "raw_lead_id": raw.json()["raw_lead"]["id"],
                        "company_name": "招银网络科技",
                        "title": "后端开发工程师",
                        "city": "深圳",
                        "job_direction": "backend",
                        "graduation_year": "2027",
                        "source_url": "https://mp.weixin.qq.com/s/example",
                        "apply_url": "https://cmbntjob.cmbchina.com",
                        "job_type": "campus",
                        "jd_text": "负责 Java 后端服务开发。",
                        "skills": ["Java", "Spring"],
                        "confidence_score": 86.5,
                    },
                )
                lead_id = lead.json()["id"]
                verified = await client.post(
                    f"/api/v1/job-leads/{lead_id}/verify",
                    json={
                        "verification_status": "verified",
                        "verified_url": "https://cmbntjob.cmbchina.com",
                        "verification_notes": "官网入口可访问。",
                    },
                )
                converted = await client.post(f"/api/v1/job-leads/{lead_id}/convert")
                return raw, lead, verified, converted

        raw_response, lead_response, verified_response, converted_response = run(call_api())

        with self.Session() as session:
            stored_job = session.scalars(select(Job)).one()

        self.assertEqual(201, raw_response.status_code)
        self.assertTrue(raw_response.json()["created"])
        self.assertEqual(201, lead_response.status_code)
        self.assertEqual("unverified", lead_response.json()["verification_status"])
        self.assertEqual(200, verified_response.status_code)
        self.assertEqual("verified", verified_response.json()["verification_status"])
        self.assertEqual(200, converted_response.status_code)
        self.assertEqual("converted", converted_response.json()["lead"]["verification_status"])
        self.assertEqual("招银网络科技", stored_job.company.name)
        self.assertEqual("job_lead", stored_job.source)

    def test_extract_job_leads_from_raw_social_content(self):
        from app.domains.jobs.models import JobLead, RawJobLead

        app = self._app(social_provider=FakeSocialLeadProvider())

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                source = await client.post(
                    "/api/v1/job-sources",
                    json={
                        "name": "xiaohongshu autumn recruiting summary",
                        "source_type": "xiaohongshu_note",
                        "entry_url": "https://www.xiaohongshu.com/discovery/item/demo",
                        "trust_level": "medium",
                        "fetch_mode": "manual_clip",
                    },
                )
                extracted = await client.post(
                    "/api/v1/job-leads/extract",
                    json={
                        "source_id": source.json()["id"],
                        "source_url": "https://www.xiaohongshu.com/discovery/item/demo",
                        "raw_content": "Xiaomi 2027 autumn recruiting Java backend Beijing Spring",
                        "content_type": "text/plain",
                    },
                )
                return extracted

        response = run(call_api())

        with self.Session() as session:
            raw_leads = session.scalars(select(RawJobLead)).all()
            stored_leads = session.scalars(select(JobLead)).all()

        self.assertEqual(201, response.status_code)
        self.assertTrue(response.json()["raw_created"])
        self.assertEqual("extracted", response.json()["raw_lead"]["status"])
        self.assertEqual(1, response.json()["extracted_count"])
        self.assertEqual("Xiaomi", response.json()["leads"][0]["company_name"])
        self.assertEqual(1, len(raw_leads))
        self.assertEqual("Xiaomi", stored_leads[0].company_name)
        self.assertEqual("unverified", stored_leads[0].verification_status)

    def test_sync_university_career_source_creates_raw_and_job_leads(self):
        from app.domains.jobs.models import JobLead, RawJobLead, SourceSyncRun

        app = self._app(
            social_provider=FakeSocialLeadProvider(),
            university_provider=FakeUniversityCareerProvider(),
        )

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                source = await client.post(
                    "/api/v1/job-sources",
                    json={
                        "name": "DLMU career site",
                        "source_type": "university_career_site",
                        "entry_url": "https://career.example.edu/jobs",
                        "trust_level": "high",
                        "fetch_mode": "public_html",
                    },
                )
                synced = await client.post(
                    f"/api/v1/job-sources/{source.json()['id']}/sync",
                    json={"limit": 5},
                )
                return synced

        response = run(call_api())

        with self.Session() as session:
            runs = session.scalars(select(SourceSyncRun)).all()
            raw_leads = session.scalars(select(RawJobLead)).all()
            stored_leads = session.scalars(select(JobLead)).all()

        self.assertEqual(200, response.status_code)
        self.assertEqual("succeeded", response.json()["status"])
        self.assertEqual(1, response.json()["fetched_count"])
        self.assertEqual(1, response.json()["extracted_count"])
        self.assertEqual(1, len(runs))
        self.assertEqual("succeeded", runs[0].status)
        self.assertEqual("extracted", raw_leads[0].status)
        self.assertEqual("Xiaomi", stored_leads[0].company_name)

    def test_lazy_verify_job_lead_converts_to_formal_job(self):
        from app.domains.jobs.models import Job

        app = self._app(lead_verifier=FakeOpenLeadVerifier())

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                source = await client.post(
                    "/api/v1/job-sources",
                    json={
                        "name": "DLMU verified source",
                        "source_type": "university_career_site",
                        "entry_url": "https://career.example.edu/jobs",
                        "trust_level": "high",
                        "fetch_mode": "public_html",
                    },
                )
                lead = await client.post(
                    "/api/v1/job-leads",
                    json={
                        "source_id": source.json()["id"],
                        "company_name": "Li Auto",
                        "title": "Backend Engineer",
                        "city": "Beijing",
                        "source_url": "https://career.example.edu/job/101",
                        "apply_url": "https://career.lixiang.com/campus/101",
                        "job_type": "campus",
                        "skills": ["Java"],
                    },
                )
                converted = await client.post(
                    f"/api/v1/job-leads/{lead.json()['id']}/verify-and-convert"
                )
                return converted

        response = run(call_api())

        with self.Session() as session:
            stored_job = session.scalars(select(Job)).one()

        self.assertEqual(200, response.status_code)
        self.assertEqual("converted", response.json()["lead"]["verification_status"])
        self.assertEqual("https://career.lixiang.com/campus/101", response.json()["lead"]["verified_url"])
        self.assertEqual("Li Auto", stored_job.company.name)
        self.assertEqual("job_lead", stored_job.source)


if __name__ == "__main__":
    unittest.main()
