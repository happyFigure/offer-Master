import sys
import unittest
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class FakeExtractor:
    def extract(self, raw_content, source_context):
        from app.domains.jobs.providers.social_lead import ExtractedJobLead

        return [
            ExtractedJobLead(
                company_name="小米集团",
                title="Java 后端开发工程师",
                city="北京",
                job_direction="backend",
                graduation_year="2027",
                source_url=source_context["source_url"],
                apply_url="https://hr.xiaomi.com/campus",
                job_type="campus",
                jd_text="参与 Java 后端服务开发。",
                skills=["Java", "Spring"],
                deadline=date(2026, 10, 31),
                confidence_score=88.0,
            ),
            ExtractedJobLead(
                company_name="蚂蚁集团",
                title="Agent 应用工程师",
                city="杭州",
                job_direction="agent_ai",
                graduation_year="2027",
                source_url=source_context["source_url"],
                apply_url="https://talent.antgroup.com/campus",
                job_type="campus",
                jd_text="建设 AI Agent 应用和工具链。",
                skills=["Python", "LLM", "Agent"],
                confidence_score=91.5,
            ),
        ]


class SocialLeadImportProviderTest(unittest.TestCase):
    def test_provider_maps_structured_extraction_to_job_lead_drafts(self):
        from app.domains.jobs.models import JobSourceTrustLevel
        from app.domains.jobs.providers.social_lead import SocialLeadImportProvider

        provider = SocialLeadImportProvider(extractor=FakeExtractor())

        drafts = provider.extract(
            source_id="source-001",
            raw_lead_id="raw-001",
            raw_content="小米、蚂蚁集团 2027 秋招开放，后端和 Agent 岗位。",
            source_url="https://www.xiaohongshu.com/discovery/item/demo",
            trust_level=JobSourceTrustLevel.MEDIUM,
        )

        self.assertEqual(2, len(drafts))
        self.assertEqual("小米集团", drafts[0].company_name)
        self.assertEqual("Java 后端开发工程师", drafts[0].title)
        self.assertEqual("2027", drafts[0].graduation_year)
        self.assertEqual(["Java", "Spring"], drafts[0].skills)
        self.assertEqual(JobSourceTrustLevel.MEDIUM, drafts[0].trust_level)
        self.assertEqual("蚂蚁集团", drafts[1].company_name)


if __name__ == "__main__":
    unittest.main()
