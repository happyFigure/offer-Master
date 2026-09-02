import sys
import unittest
import warnings
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


class FakeWeChatArticleFetcher:
    tool_name = "WeChatArticleFetcher"

    def fetch(self, url, context, *, domain_health=None):
        from app.domains.jobs.schemas import ToolResult, ToolSuggestedNextAction

        text = "信息来源：公众号：腾讯招聘\n有鹅选鹅！腾讯2027校园招聘全球启动"
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
                "url": url,
                "final_url": url,
                "status_code": 200,
                "title": "【招聘】腾讯2027届校园招聘",
                "text": text,
                "content_length": len(text),
                "candidate_links": ["https://mp.weixin.qq.com/s/tencent-campus-child"],
                "extraction_method": "fake_wechat_html",
            },
        )


class FakeEmptyJobLeadProvider:
    def extract(self, source_id, raw_lead_id, raw_content, source_url, trust_level):
        return []


class FakeWeChatAccountProvider:
    def discover(self, source, limit):
        from app.agent_runtime.workflows.job_discovery import WeChatAccountArticleEntry

        return [
            WeChatAccountArticleEntry(
                title="【招聘】腾讯2027届校园招聘",
                url="https://mp.weixin.qq.com/s/tencent-campus",
                source_account="大连海事就业",
                raw_payload={"read_count": 474},
            ),
            WeChatAccountArticleEntry(
                title="【招聘】字节跳动2027届校园招聘",
                url="https://mp.weixin.qq.com/s/bytedance-campus",
                source_account="大连海事就业",
                raw_payload={"read_count": 2398},
            ),
        ]


class WeChatRecruitingSignalFlowTest(unittest.TestCase):
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

    def test_wechat_article_url_import_saves_recruiting_signal_when_no_job_leads_exist(self):
        from app.agent_runtime.workflows.url_import import (
            UrlImportCommand,
            build_url_import_dependencies,
            run_url_import_workflow,
        )
        from app.domains.jobs.models import RecruitingSignal

        with self.Session() as session:
            result = run_url_import_workflow(
                UrlImportCommand(
                    url="https://mp.weixin.qq.com/s/eRjq-YMOhENXvbvYzZynpg",
                    source_hint="wechat_article",
                ),
                dependencies=build_url_import_dependencies(
                    session,
                    fetchers={"WeChatArticleFetcher": FakeWeChatArticleFetcher()},
                    social_provider=FakeEmptyJobLeadProvider(),
                ),
            )
            session.commit()

            signals = session.scalars(select(RecruitingSignal)).all()

        self.assertEqual("partial", result.url_import_run.status)
        self.assertEqual("extract_recruiting_signals", result.url_import_run.current_stage)
        self.assertEqual("enrich_recruiting_signal", result.url_import_run.next_action)
        self.assertEqual(1, result.url_import_run.run_metadata["recruiting_signal_count"])
        self.assertEqual(1, len(signals))
        self.assertEqual("腾讯", signals[0].company_name)
        self.assertEqual("campus_recruitment_open", signals[0].signal_type)
        self.assertEqual("2027", signals[0].graduation_year)
        self.assertEqual("needs_job_enrichment", signals[0].status)

    def test_wechat_account_sync_stores_article_candidates_without_university_provider(self):
        from app.agent_runtime.workflows.job_discovery import (
            WeChatAccountSyncCommand,
            run_wechat_account_source_sync,
        )
        from app.domains.jobs.models import ArticleCandidate, JobSourceFetchMode, JobSourceTrustLevel, JobSourceType, RecruitingSignal
        from app.domains.jobs.repository import (
            ArticleCandidateRepository,
            JobLeadRepository,
            JobSourceRepository,
            RawJobLeadRepository,
            RecruitingSignalRepository,
            SourceSyncRunRepository,
        )
        from app.domains.jobs.schemas import JobSourceCreate
        from app.domains.jobs.service import JobLeadService

        with self.Session() as session:
            lead_service = JobLeadService(
                sources=JobSourceRepository(session),
                sync_runs=SourceSyncRunRepository(session),
                raw_leads=RawJobLeadRepository(session),
                leads=JobLeadRepository(session),
                article_candidates=ArticleCandidateRepository(session),
                recruiting_signals=RecruitingSignalRepository(session),
            )
            source = lead_service.create_source(
                JobSourceCreate(
                    name="大连海事就业",
                    source_type=JobSourceType.WECHAT_ACCOUNT,
                    fetch_mode=JobSourceFetchMode.MCP_VISIBLE_PAGE,
                    trust_level=JobSourceTrustLevel.HIGH,
                    sync_interval_hours=24,
                    notes="公众号账号，按文章列表同步候选招聘文章。",
                )
            )
            result = run_wechat_account_source_sync(
                WeChatAccountSyncCommand(source_id=source.id, limit=20),
                lead_service=lead_service,
                article_provider=FakeWeChatAccountProvider(),
            )
            session.commit()

            candidates = session.scalars(select(ArticleCandidate).order_by(ArticleCandidate.title)).all()
            signals = session.scalars(select(RecruitingSignal).order_by(RecruitingSignal.company_name)).all()

        self.assertEqual("succeeded", result.sync_run.status)
        self.assertEqual(2, result.fetched_count)
        self.assertEqual(2, len(candidates))
        self.assertEqual(2, len(signals))
        self.assertEqual({"pending"}, {candidate.status for candidate in candidates})
        self.assertEqual({"大连海事就业"}, {candidate.source_account for candidate in candidates})
        self.assertEqual({"腾讯", "字节跳动"}, {signal.company_name for signal in signals})
        self.assertEqual({"2027"}, {signal.graduation_year for signal in signals})
        self.assertEqual({"needs_job_enrichment"}, {signal.status for signal in signals})
        self.assertEqual({candidate.id for candidate in candidates}, {signal.article_candidate_id for signal in signals})


if __name__ == "__main__":
    unittest.main()
