import sys
import unittest
from asyncio import run
from pathlib import Path

from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class JobBoardApiTest(unittest.TestCase):
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

    def test_offerio_companies_endpoint_returns_temporary_external_data(self):
        from app.api.v1.jobs import get_offerio_provider
        from app.db.session import get_db_session
        from app.domains.jobs.providers.offerio import OfferIOCompany, OfferIOPage
        from app.main import create_app

        class FakeOfferIOProvider:
            def list_companies(self, **kwargs):
                self.kwargs = kwargs
                return OfferIOPage(
                    items=[
                        OfferIOCompany(
                            name="腾讯",
                            company_nature="民企",
                            industry="互联网/游戏/软件",
                            locations="深圳总部、上海、北京",
                            job_count=98,
                            updated_at="2026-07-29",
                            raw_payload={"company": "腾讯"},
                        )
                    ],
                    page=1,
                    page_size=20,
                    total=76,
                    total_pages=4,
                )

        provider = FakeOfferIOProvider()
        app = create_app()

        def override_session():
            with self.Session() as session:
                yield session

        app.dependency_overrides[get_db_session] = override_session
        app.dependency_overrides[get_offerio_provider] = lambda: provider

        async def request():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/api/v1/jobs/offerio/companies?job_type=校招&page=1&page_size=20")

        response = run(request())

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(76, payload["total"])
        self.assertEqual("腾讯", payload["items"][0]["name"])
        self.assertEqual(98, payload["items"][0]["job_count"])
        self.assertEqual("校招", provider.kwargs["job_type"])

    def test_offerio_company_openings_endpoint_returns_paginated_openings(self):
        from app.api.v1.jobs import get_offerio_provider
        from app.db.session import get_db_session
        from app.domains.jobs.providers.offerio import OfferIOCompanyOpening, OfferIOPage
        from app.main import create_app

        class FakeOfferIOProvider:
            def list_company_openings(self, **kwargs):
                self.kwargs = kwargs
                return OfferIOPage(
                    items=[
                        OfferIOCompanyOpening(
                            id="opening-001",
                            company_name="Tencent",
                            company_nature="private",
                            industry="internet/software",
                            batch="autumn",
                            target="2027",
                            location="Shenzhen",
                            positions="Java backend, AI agent platform",
                            update_date="2026/08/17",
                            deadline="until filled",
                            apply_link="https://join.qq.com/",
                            has_written_test="written test required",
                            raw_payload={"id": "opening-001"},
                        )
                    ],
                    page=1,
                    page_size=50,
                    total=3506,
                    total_pages=71,
                )

        provider = FakeOfferIOProvider()
        app = create_app()

        def override_session():
            with self.Session() as session:
                yield session

        app.dependency_overrides[get_db_session] = override_session
        app.dependency_overrides[get_offerio_provider] = lambda: provider

        async def request():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/api/v1/jobs/offerio/company-openings?page=1&page_size=50&keyword=Java&industry=internet")

        response = run(request())

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(3506, payload["total"])
        self.assertEqual(50, payload["page_size"])
        self.assertEqual("Tencent", payload["items"][0]["company_name"])
        self.assertEqual("Java backend, AI agent platform", payload["items"][0]["positions"])
        self.assertEqual("Java", provider.kwargs["keyword"])
        self.assertEqual("internet", provider.kwargs["industry"])

    def test_application_board_can_create_from_job_payload_and_update_stage(self):
        from app.db.session import get_db_session
        from app.domains.applications.models import ApplicationStatus
        from app.main import create_app

        app = create_app()

        def override_session():
            with self.Session() as session:
                yield session

        app.dependency_overrides[get_db_session] = override_session

        async def request_flow():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                created = await client.post(
                    "/api/v1/applications/from-job",
                    json={
                        "job": {
                            "company_name": "腾讯",
                            "company_industry": "互联网/游戏/软件",
                            "title": "后台开发工程师",
                            "city": "深圳总部",
                            "source": "offerio",
                            "source_job_id": "tencent_backend_001",
                            "source_url": "https://join.qq.com/post_detail.html?postid=tencent_backend_001",
                            "job_type": "校招",
                            "jd_text": "负责 Java 后端服务开发。",
                            "skills": ["Java", "Spring"],
                        },
                        "status": "evaluating",
                        "priority": "high",
                        "channel": "offerio",
                        "notes": "先放入待投递看板。",
                    },
                )
                application_id = created.json()["id"]
                updated = await client.patch(
                    f"/api/v1/applications/{application_id}",
                    json={"status": "applied", "notes": "已经投递，等待反馈。"},
                )
                listed = await client.get("/api/v1/applications")
                return created, updated, listed

        created, updated, listed = run(request_flow())

        self.assertEqual(200, created.status_code)
        self.assertEqual(ApplicationStatus.EVALUATING, created.json()["status"])
        self.assertEqual("腾讯", created.json()["job"]["company"]["name"])
        self.assertEqual(200, updated.status_code)
        self.assertEqual(ApplicationStatus.APPLIED, updated.json()["status"])
        self.assertEqual("已经投递，等待反馈。", updated.json()["notes"])
        self.assertEqual(200, listed.status_code)
        self.assertEqual(1, len(listed.json()["items"]))
        self.assertEqual("后台开发工程师", listed.json()["items"][0]["job"]["title"])


if __name__ == "__main__":
    unittest.main()
