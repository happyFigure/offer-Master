from pathlib import Path
from unittest import TestCase


class FrontendRecruitingSignalShowcaseTest(TestCase):
    def test_company_exhibition_does_not_surface_incomplete_article_company_sources(self) -> None:
        app_source = Path("apps/web/src/app/App.tsx").read_text(encoding="utf-8")

        imported_leads_index = app_source.find("<ImportedLeadList leads={importedLeads}")

        self.assertNotEqual(-1, imported_leads_index)
        self.assertNotIn("<ArticleCompanySourceList signals={recruitingSignals}", app_source)
        self.assertNotIn('title="文章/社媒公司来源"', app_source)
        self.assertNotIn("dedupeRecruitingSignalsForJobBoard", app_source)
