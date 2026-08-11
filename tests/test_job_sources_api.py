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

    def _app(self):
        from app.db.session import get_db_session
        from app.main import create_app

        app = create_app()

        def override_session():
            with self.Session() as session:
                yield session

        app.dependency_overrides[get_db_session] = override_session
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


if __name__ == "__main__":
    unittest.main()
