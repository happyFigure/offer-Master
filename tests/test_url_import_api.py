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


class FakeUrlImportFetcher:
    tool_name = "HTTPArticleFetcher"

    def __init__(self):
        self.calls = 0

    def fetch(self, url, context, *, domain_health=None):
        from app.domains.jobs.schemas import ToolResult, ToolSuggestedNextAction

        self.calls += 1
        text = (
            "JD.com 2027 campus recruiting opens Java backend engineer and Agent platform "
            "engineering roles in Beijing. Apply through the official career page."
        )
        return ToolResult(
            ok=True,
            stage=context.stage,
            tool_name=self.tool_name,
            suggested_next_action=ToolSuggestedNextAction.CONTINUE_WORKFLOW,
            cost={
                "tool_calls": context.tool_call_count + 1,
                "llm_calls": context.llm_call_count,
                "fetch_attempts_for_stage": context.fetch_attempts_for_stage + 1,
                "mcp_requests": context.mcp_request_count,
            },
            artifacts={
                "final_url": url,
                "status_code": 200,
                "title": "JD 2027 Campus Recruiting",
                "text": text,
                "content_length": len(text),
                "candidate_links": ["https://career.example.com/apply/java-backend"],
                "extraction_method": "fake_http",
            },
        )


class FakeUrlImportSocialProvider:
    def __init__(self):
        self.calls = 0

    def extract(self, source_id, raw_lead_id, raw_content, source_url, trust_level):
        from app.domains.jobs.schemas import JobLeadCreate

        self.calls += 1
        return [
            JobLeadCreate(
                source_id=source_id,
                raw_lead_id=raw_lead_id,
                company_name="JD.com",
                title="Java Backend Engineer",
                city="Beijing",
                job_direction="backend",
                graduation_year="2027",
                source_url=source_url,
                apply_url="https://career.example.com/apply/java-backend",
                job_type="campus",
                jd_text="Java backend and Agent workflow platform engineering.",
                skills=["Java", "Spring", "Agent"],
                confidence_score=86,
                trust_level=trust_level,
            )
        ]


class UrlImportApiTest(unittest.TestCase):
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
        self.fetcher = FakeUrlImportFetcher()
        self.provider = FakeUrlImportSocialProvider()

    def tearDown(self):
        self.engine.dispose()

    def _app(self):
        from app.api.v1.job_sources import (
            get_social_lead_provider,
            get_url_import_fetchers,
            get_url_import_session_factory,
        )
        from app.db.session import get_db_session
        from app.main import create_app

        app = create_app()

        def override_session():
            with self.Session() as session:
                yield session

        app.dependency_overrides[get_db_session] = override_session
        app.dependency_overrides[get_social_lead_provider] = lambda: self.provider
        app.dependency_overrides[get_url_import_fetchers] = lambda: {
            "HTTPArticleFetcher": self.fetcher
        }
        app.dependency_overrides[get_url_import_session_factory] = lambda: self.Session
        return app

    def test_import_url_creates_run_and_query_returns_completed_result(self):
        from app.domains.jobs.models import JobLead, RawJobLead, UrlImportRun

        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                created = await client.post(
                    "/api/v1/job-leads/import-url",
                    json={"url": "https://career.example.com/jobs/java?utm_source=xhs"},
                )
                run_id = created.json()["run_id"]
                queried = await client.get(f"/api/v1/job-leads/import-runs/{run_id}")
                return created, queried

        created_response, queried_response = run(call_api())

        with self.Session() as session:
            url_runs = session.scalars(select(UrlImportRun)).all()
            raw_leads = session.scalars(select(RawJobLead)).all()
            job_leads = session.scalars(select(JobLead)).all()

        self.assertEqual(202, created_response.status_code)
        self.assertEqual("running", created_response.json()["status"])
        self.assertEqual("queued", created_response.json()["current_stage"])
        self.assertEqual("unknown", created_response.json()["domain_health_state"])
        self.assertEqual(200, queried_response.status_code)
        self.assertEqual("succeeded", queried_response.json()["status"])
        self.assertEqual("completed", queried_response.json()["current_stage"])
        self.assertEqual("https://career.example.com/jobs/java", queried_response.json()["normalized_url"])
        self.assertEqual(1, queried_response.json()["extracted_count"])
        self.assertEqual(1, self.fetcher.calls)
        self.assertEqual(1, self.provider.calls)
        self.assertEqual(1, len(url_runs))
        self.assertEqual(1, len(raw_leads))
        self.assertEqual(1, len(job_leads))
        self.assertEqual("JD.com", job_leads[0].company_name)

    def test_import_run_query_returns_404_for_missing_run(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/api/v1/job-leads/import-runs/missing-run-id")

        response = run(call_api())

        self.assertEqual(404, response.status_code)
        self.assertIn("URL import run not found", response.json()["detail"])

    def test_xiaohongshu_import_waits_for_user_visible_page_without_background_fetch(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                created = await client.post(
                    "/api/v1/job-leads/import-url",
                    json={"url": "https://www.xiaohongshu.com/discovery/item/demo?utm_source=share"},
                )
                queried = await client.get(
                    f"/api/v1/job-leads/import-runs/{created.json()['run_id']}"
                )
                return created, queried

        created_response, queried_response = run(call_api())

        self.assertEqual(202, created_response.status_code)
        self.assertEqual("waiting_user", created_response.json()["status"])
        self.assertEqual(200, queried_response.status_code)
        self.assertEqual("waiting_user", queried_response.json()["status"])
        self.assertEqual("REQUIRES_MCP_VISIBLE_PAGE", queried_response.json()["error_code"])
        self.assertEqual("request_user_visible_page", queried_response.json()["next_action"])
        self.assertEqual(0, self.fetcher.calls)
        self.assertEqual(0, self.provider.calls)

    def test_xiaohongshu_visible_page_content_resumes_waiting_import_and_saves_note_text(self):
        from app.domains.jobs.models import RawJobLead, UrlImportRun

        app = self._app()
        visible_text = """
登录后推荐更懂你的笔记
1/3
悲伤土豆鸡肉饭
关注
27 届秋招 | 七月总结
最近陆续有一些公司开了秋招，但整体来说开的数量还不算特别多。
米哈游 秋招          7.22 一面  7.28 二面（已挂）
百度                        7.23 笔试
07-31 四川
共 37 条评论
登录查看全部评论内容
"""

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                created = await client.post(
                    "/api/v1/job-leads/import-url",
                    json={"url": "https://www.xiaohongshu.com/discovery/item/demo"},
                )
                resumed = await client.post(
                    f"/api/v1/job-leads/import-runs/{created.json()['run_id']}/visible-page-content",
                    json={
                        "title": "27 届秋招 | 七月总结 - 小红书",
                        "final_url": "https://www.xiaohongshu.com/explore/demo",
                        "visible_text": visible_text,
                    },
                )
                queried = await client.get(
                    f"/api/v1/job-leads/import-runs/{created.json()['run_id']}"
                )
                return created, resumed, queried

        created_response, resumed_response, queried_response = run(call_api())

        with self.Session() as session:
            url_run = session.scalars(select(UrlImportRun)).one()
            raw_lead = session.scalars(select(RawJobLead)).one()

        self.assertEqual(202, created_response.status_code)
        self.assertEqual("waiting_user", created_response.json()["status"])
        self.assertEqual(200, resumed_response.status_code)
        self.assertEqual(200, queried_response.status_code)
        self.assertIsNotNone(url_run.raw_job_lead_id)
        self.assertIn("27 届秋招 | 七月总结", raw_lead.raw_content)
        self.assertIn("米哈游 秋招", raw_lead.raw_content)
        self.assertNotIn("登录后推荐", raw_lead.raw_content)
        self.assertEqual("xiaohongshu_visible_text", raw_lead.raw_payload["extraction_method"])
        self.assertEqual(3, raw_lead.raw_payload["image_count"])
        self.assertTrue(raw_lead.raw_payload["image_parse_deferred"])

        resumed_body = resumed_response.json()
        queried_body = queried_response.json()
        self.assertEqual(raw_lead.raw_content[:500], resumed_body["raw_content_preview"])
        self.assertEqual(resumed_body["raw_content_preview"], queried_body["raw_content_preview"])
        self.assertEqual("xiaohongshu_visible_text", resumed_body["raw_extraction_method"])
        self.assertEqual(3, resumed_body["raw_image_count"])
        self.assertTrue(resumed_body["raw_image_parse_deferred"])

    def test_domain_health_endpoints_list_all_and_filter_by_domain(self):
        app = self._app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                await client.post(
                    "/api/v1/job-leads/import-url",
                    json={"url": "https://career.example.com/jobs/java"},
                )
                all_domains = await client.get("/api/v1/tool-health/domains")
                one_domain = await client.get("/api/v1/tool-health/domains/career.example.com")
                return all_domains, one_domain

        all_response, one_response = run(call_api())

        self.assertEqual(200, all_response.status_code)
        self.assertEqual(200, one_response.status_code)
        self.assertEqual("career.example.com", all_response.json()["items"][0]["domain"])
        self.assertEqual("HTTPArticleFetcher", all_response.json()["items"][0]["tool_name"])
        self.assertEqual("career.example.com", one_response.json()["items"][0]["domain"])


if __name__ == "__main__":
    unittest.main()
