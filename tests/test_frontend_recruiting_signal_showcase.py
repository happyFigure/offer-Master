from pathlib import Path
from unittest import TestCase


class FrontendRecruitingSignalShowcaseTest(TestCase):
    def test_job_exhibition_surfaces_article_company_sources_after_imported_leads(self) -> None:
        app_source = Path("apps/web/src/app/App.tsx").read_text(encoding="utf-8")

        imported_leads_index = app_source.find("<ImportedLeadList leads={importedLeads}")
        company_sources_index = app_source.find("<ArticleCompanySourceList signals={recruitingSignals}")

        self.assertNotEqual(-1, imported_leads_index)
        self.assertNotEqual(-1, company_sources_index)
        self.assertLess(imported_leads_index, company_sources_index)
        self.assertIn('title="文章/社媒公司来源"', app_source)
        self.assertIn("dedupeRecruitingSignalsForJobBoard", app_source)
