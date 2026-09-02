import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FrontendJobBoardUiTest(unittest.TestCase):
    def test_app_exposes_job_exhibition_and_application_progress_pages(self):
        app_source = (PROJECT_ROOT / "apps" / "web" / "src" / "app" / "App.tsx").read_text(encoding="utf-8")

        self.assertIn('"jobs"', app_source)
        self.assertIn("公司展览", app_source)
        self.assertIn("查看岗位", app_source)
        self.assertIn("加入投递板", app_source)
        self.assertIn("投递进度", app_source)
        self.assertIn("待投递", app_source)
        self.assertIn("已投未回", app_source)
        self.assertIn("测评/AI面试", app_source)
        self.assertIn("一面", app_source)

    def test_company_exhibition_has_source_classification_and_pagination_controls(self):
        app_source = (PROJECT_ROOT / "apps" / "web" / "src" / "app" / "App.tsx").read_text(encoding="utf-8")
        job_board_api = (PROJECT_ROOT / "apps" / "web" / "src" / "api" / "jobBoard.ts").read_text(encoding="utf-8")
        jobs_types = (PROJECT_ROOT / "apps" / "web" / "src" / "types" / "jobs.ts").read_text(encoding="utf-8")

        self.assertIn("开放岗位公司库", app_source)
        self.assertIn("公司聚合岗位库", app_source)
        self.assertIn("每页", app_source)
        self.assertIn("上一页", app_source)
        self.assertIn("下一页", app_source)
        self.assertIn("OfferIOCompanyOpening", jobs_types)
        self.assertIn("listOfferIOCompanyOpenings", job_board_api)
        self.assertIn("/api/v1/jobs/offerio/company-openings", job_board_api)

    def test_company_exhibition_requests_use_longer_timeout_than_generic_api(self):
        job_board_api = (PROJECT_ROOT / "apps" / "web" / "src" / "api" / "jobBoard.ts").read_text(encoding="utf-8")

        self.assertIn("const OFFERIO_SOURCE_TIMEOUT_MS = 25_000", job_board_api)
        self.assertIn("{ timeoutMs: OFFERIO_SOURCE_TIMEOUT_MS }", job_board_api)

    def test_job_board_metric_labels_make_count_scope_explicit(self):
        app_source = (PROJECT_ROOT / "apps" / "web" / "src" / "app" / "App.tsx").read_text(encoding="utf-8")

        self.assertIn('label="来源库公司数"', app_source)
        self.assertIn('label="当前筛选导入线索"', app_source)
        self.assertNotIn('label="当前筛选社媒/文章公司"', app_source)
        self.assertNotIn("来源库岗位数", app_source)

    def test_company_exhibition_excludes_incomplete_article_social_company_signals(self):
        app_source = (PROJECT_ROOT / "apps" / "web" / "src" / "app" / "App.tsx").read_text(encoding="utf-8")

        self.assertIn("<ImportedLeadList leads={importedLeads}", app_source)
        self.assertNotIn("<ArticleCompanySourceList signals={recruitingSignals}", app_source)
        self.assertNotIn("dedupeRecruitingSignalsForJobBoard", app_source)
        self.assertIn("normalizeCompanyNameForDedupe", app_source)

    def test_offerio_company_source_sync_uses_multi_page_total_limit(self):
        app_source = (PROJECT_ROOT / "apps" / "web" / "src" / "app" / "App.tsx").read_text(encoding="utf-8")

        self.assertIn("syncLimitForSource", app_source)
        self.assertIn("/api/recruitment/job-companies", app_source)
        self.assertIn("syncJobSource(source.id, syncLimitForSource(source))", app_source)
        self.assertIn("return 1000", app_source)

    def test_frontend_has_typed_job_board_and_application_apis(self):
        job_board_api = PROJECT_ROOT / "apps" / "web" / "src" / "api" / "jobBoard.ts"
        applications_api = PROJECT_ROOT / "apps" / "web" / "src" / "api" / "applications.ts"
        jobs_types = (PROJECT_ROOT / "apps" / "web" / "src" / "types" / "jobs.ts").read_text(encoding="utf-8")

        self.assertTrue(job_board_api.is_file())
        self.assertTrue(applications_api.is_file())
        self.assertIn("OfferIOCompany", jobs_types)
        self.assertIn("ApplicationBoardItem", jobs_types)
        self.assertIn("ApplicationStatus", jobs_types)


if __name__ == "__main__":
    unittest.main()
