from pathlib import Path
from unittest import TestCase


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FrontendLeadsBoundaryUiTest(TestCase):
    def test_leads_page_is_import_only_and_company_exhibition_excludes_incomplete_signals(self) -> None:
        app_source = (PROJECT_ROOT / "apps" / "web" / "src" / "app" / "App.tsx").read_text(encoding="utf-8")

        self.assertIn('leads: "线索导入"', app_source)
        self.assertIn('jobs: "公司展览"', app_source)
        self.assertNotIn('title="文章/社媒公司来源"', app_source)
        self.assertIn('title="导入岗位线索"', app_source)
        self.assertIn('title="临时链接导入"', app_source)
        self.assertIn("URL 粘贴解析只是一次性兜底", app_source)
        self.assertNotIn("dedupeRecruitingSignalsForJobBoard", app_source)
        self.assertIn("normalizeCompanyNameForDedupe", app_source)
        self.assertNotIn('title="待补全岗位线索"', app_source)
        self.assertNotIn('title="公司开放信号"', app_source)
        self.assertNotIn('title="具体岗位线索"', app_source)
        self.assertNotIn('title="已发现秋招公司"', app_source)
