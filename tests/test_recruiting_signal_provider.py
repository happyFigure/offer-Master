import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class RecruitingSignalProviderTest(unittest.TestCase):
    def test_provider_ignores_navigation_collection_lines(self):
        from app.domains.jobs.providers.recruiting_signal import RuleBasedRecruitingSignalProvider

        raw_content = (
            "招聘信息合集推荐 交通运输部部属单位 国家24365大学生就业服务 "
            "选调生/公务员 事业单位 军队文职 就业政策 实习信息 宣讲信息 "
            "2027届校园招聘 2026届校园招聘 （请点击查看合集）"
        )

        drafts = RuleBasedRecruitingSignalProvider().extract(
            source_id="source-1",
            raw_lead_id=None,
            raw_content=raw_content,
            source_url="https://mp.weixin.qq.com/s/example",
            trust_level="high",
        )

        self.assertEqual([], drafts)


if __name__ == "__main__":
    unittest.main()
