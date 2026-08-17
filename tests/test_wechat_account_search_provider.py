import sys
import unittest
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class WeChatAccountSearchProviderTest(unittest.TestCase):
    def test_discover_extracts_recruiting_articles_from_public_search_results(self):
        from app.domains.jobs.models import JobSource, JobSourceFetchMode, JobSourceTrustLevel, JobSourceType
        from app.domains.jobs.providers.wechat_account_search import WeChatAccountSearchProvider

        search_html = """
        <html><body>
          <ul class="news-list">
            <li>
              <h3><a href="/link?url=tencent">【招聘】腾讯2027届校园招聘</a></h3>
              <a class="account">大连海事就业</a>
              <span class="s2">今天</span>
            </li>
            <li>
              <h3><a href="/link?url=policy">就业政策说明</a></h3>
              <a class="account">大连海事就业</a>
            </li>
            <li>
              <h3><a href="https://mp.weixin.qq.com/s/bytedance-campus">【招聘】字节跳动2027届校园招聘</a></h3>
              <a class="account">大连海事就业</a>
              <span class="s2">昨天</span>
            </li>
          </ul>
        </body></html>
        """

        def handler(request):
            url = str(request.url)
            if url.startswith("https://weixin.sogou.com/weixin?"):
                return httpx.Response(200, text=search_html)
            if url == "https://weixin.sogou.com/link?url=tencent":
                return httpx.Response(302, headers={"Location": "https://mp.weixin.qq.com/s/tencent-campus"})
            return httpx.Response(404)

        provider = WeChatAccountSearchProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
        source = JobSource(
            id="source-1",
            name="大连海事就业",
            source_type=JobSourceType.WECHAT_ACCOUNT,
            fetch_mode=JobSourceFetchMode.MCP_VISIBLE_PAGE,
            trust_level=JobSourceTrustLevel.HIGH,
            sync_interval_hours=24,
        )

        entries = provider.discover(source, limit=10)

        self.assertEqual(2, len(entries))
        self.assertEqual("【招聘】腾讯2027届校园招聘", entries[0].title)
        self.assertEqual("https://mp.weixin.qq.com/s/tencent-campus", entries[0].url)
        self.assertEqual("大连海事就业", entries[0].source_account)
        self.assertEqual("sogou_weixin_search", entries[0].raw_payload["discovery_method"])
        self.assertEqual("【招聘】字节跳动2027届校园招聘", entries[1].title)

    def test_discover_tries_employment_account_alias_for_university_public_account_name(self):
        from app.domains.jobs.models import JobSource, JobSourceFetchMode, JobSourceTrustLevel, JobSourceType
        from app.domains.jobs.providers.wechat_account_search import WeChatAccountSearchProvider

        irrelevant_html = """
        <html><body><ul class="news-list"><li>
          <h3><a href="/link?url=faculty">校园招聘 | 大连海事大学2023年秋季指导员公开招聘公告</a></h3>
          <a class="account">辽宁省考信息网</a>
        </li><li>
          <h3><a href="/link?url=public-platform">【招聘】国家大学生就业服务平台招聘活动</a></h3>
          <a class="account">大连海事就业</a>
        </li></ul></body></html>
        """
        employment_html = """
        <html><body><ul class="news-list"><li>
          <h3><a href="https://mp.weixin.qq.com/s/jd-campus">【招聘】京东2027届校园招聘</a></h3>
          <a class="account">大连海事就业</a>
        </li></ul></body></html>
        """

        def handler(request):
            query = str(request.url)
            if "%E5%A4%A7%E8%BF%9E%E6%B5%B7%E4%BA%8B%E5%B0%B1%E4%B8%9A" in query:
                return httpx.Response(200, text=employment_html)
            return httpx.Response(200, text=irrelevant_html)

        provider = WeChatAccountSearchProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
        source = JobSource(
            id="source-2",
            name="大连海事大学公众号",
            source_type=JobSourceType.WECHAT_ACCOUNT,
            fetch_mode=JobSourceFetchMode.MCP_VISIBLE_PAGE,
            trust_level=JobSourceTrustLevel.HIGH,
            sync_interval_hours=24,
        )

        entries = provider.discover(source, limit=5)

        self.assertEqual(1, len(entries))
        self.assertEqual("【招聘】京东2027届校园招聘", entries[0].title)
        self.assertEqual("大连海事就业", entries[0].source_account)


if __name__ == "__main__":
    unittest.main()
