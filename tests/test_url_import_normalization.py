import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class UrlImportNormalizationTest(unittest.TestCase):
    def test_normalize_import_url_removes_tracking_params_but_preserves_ambiguous_params(self):
        from app.domains.jobs.url_import import normalize_import_url

        result = normalize_import_url(
            " HTTPS://MP.WEIXIN.QQ.COM:443/s/example?"
            "utm_source=xhs&spm=abc&from=singlemessage&share=1&"
            "source=timeline&sid=abc&sessionid=def&a=1#wechat_redirect "
        )

        self.assertEqual(
            "https://mp.weixin.qq.com/s/example?a=1&sessionid=def&sid=abc&source=timeline",
            result.normalized_url,
        )
        self.assertEqual(
            "bd09e17b69b059cd822c4105fa757effa04cb2869ac02a2fd7be022b125594f1",
            result.normalized_url_hash,
        )
        self.assertEqual("mp.weixin.qq.com", result.domain)
        self.assertEqual(["from", "share", "spm", "utm_source"], result.removed_query_keys)
        self.assertEqual(["a", "sessionid", "sid", "source"], result.preserved_query_keys)

    def test_tracking_only_variants_share_the_same_normalized_hash(self):
        from app.domains.jobs.url_import import normalize_import_url

        first = normalize_import_url("https://mp.weixin.qq.com/s/example?utm_source=xhs&spm=abc")
        second = normalize_import_url(
            "https://mp.weixin.qq.com/s/example?from=timeline&share=1&utm_medium=social"
        )

        self.assertEqual("https://mp.weixin.qq.com/s/example", first.normalized_url)
        self.assertEqual("https://mp.weixin.qq.com/s/example", second.normalized_url)
        self.assertEqual(first.normalized_url_hash, second.normalized_url_hash)
        self.assertEqual(
            "e9514928359e93fd9c27339ff466a09335eeebc44e8d9215843ab0145f9b7627",
            first.normalized_url_hash,
        )

    def test_classify_import_url_routes_wechat_xiaohongshu_boss_university_official_and_public(self):
        from app.domains.jobs.models import JobSourceFetchMode, JobSourceType
        from app.domains.jobs.url_import import analyze_import_url

        cases = [
            (
                "https://mp.weixin.qq.com/s/example",
                JobSourceType.WECHAT_ARTICLE,
                JobSourceFetchMode.PUBLIC_HTML,
                "wechat_article",
                False,
            ),
            (
                "https://www.xiaohongshu.com/discovery/item/6a77468a000000002403f1d6?source=webshare",
                JobSourceType.XIAOHONGSHU_NOTE,
                JobSourceFetchMode.MCP_VISIBLE_PAGE,
                "mcp_visible_page",
                True,
            ),
            (
                "https://www.zhipin.com/job_detail/example.html",
                JobSourceType.JOB_BOARD_VISIBLE_PAGE,
                JobSourceFetchMode.MCP_VISIBLE_PAGE,
                "mcp_visible_page",
                True,
            ),
            (
                "https://myjob.dlmu.edu.cn/job/search/d_category%5B0%5D/0",
                JobSourceType.UNIVERSITY_CAREER_SITE,
                JobSourceFetchMode.PUBLIC_HTML,
                "university_career_site",
                False,
            ),
            (
                "https://campus.jd.com/#/jobs",
                JobSourceType.OFFICIAL_CAREER_SITE,
                JobSourceFetchMode.PUBLIC_HTML,
                "official_career_site",
                False,
            ),
            (
                "https://example.com/articles/2027-campus-recruiting-java",
                JobSourceType.PUBLIC_ARTICLE,
                JobSourceFetchMode.PUBLIC_HTML,
                "http_article",
                False,
            ),
        ]

        for url, source_type, fetch_mode, fetch_layer, requires_user_confirmation in cases:
            with self.subTest(url=url):
                result = analyze_import_url(url)
                self.assertEqual(source_type, result.source_type)
                self.assertEqual(fetch_mode, result.fetch_mode)
                self.assertEqual(fetch_layer, result.fetch_layer)
                self.assertEqual(requires_user_confirmation, result.requires_user_confirmation)

    def test_url_import_run_repository_finds_duplicate_by_normalized_hash_before_fetch(self):
        from app.db.base import Base
        from app.domains.applications import models as application_models  # noqa: F401
        from app.domains.automation.models import WorkflowRun, WorkflowRunStatus
        from app.domains.jobs.models import UrlImportRun, UrlImportRunStatus
        from app.domains.jobs.repository import UrlImportRunRepository

        engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)

        with Session() as session:
            workflow = WorkflowRun(
                id="workflow-run-url-import-duplicate",
                workflow_type="url_import",
                status=WorkflowRunStatus.RUNNING,
                current_step="url_normalization",
            )
            existing = UrlImportRun(
                workflow_run_id="workflow-run-url-import-duplicate",
                input_url="https://mp.weixin.qq.com/s/example?utm_source=xhs",
                normalized_url="https://mp.weixin.qq.com/s/example",
                normalized_url_hash="e9514928359e93fd9c27339ff466a09335eeebc44e8d9215843ab0145f9b7627",
                domain="mp.weixin.qq.com",
                status=UrlImportRunStatus.SUCCEEDED,
                current_stage="completed",
            )
            session.add_all([workflow, existing])
            session.commit()

            duplicate = UrlImportRunRepository(session).get_by_normalized_url_hash(
                "e9514928359e93fd9c27339ff466a09335eeebc44e8d9215843ab0145f9b7627"
            )

        self.assertIsNotNone(duplicate)
        self.assertEqual(existing.id, duplicate.id)


if __name__ == "__main__":
    unittest.main()
