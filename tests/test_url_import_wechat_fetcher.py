import sys
import unittest
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class UrlImportWeChatFetcherTest(unittest.TestCase):
    def test_wechat_article_fetcher_extracts_title_author_publish_time_and_text(self):
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolSuggestedNextAction
        from app.domains.jobs.tool_guard import ToolCallContext
        from app.domains.jobs.wechat_fetcher import WeChatArticleFetcher

        html = """
        <html>
          <head><meta property="og:title" content="Fallback title"></head>
          <body>
            <h1 id="activity-name">2027 Autumn Recruiting Summary</h1>
            <a id="js_name">DLMU Career</a>
            <em id="publish_time">2026-08-10</em>
            <div id="js_content">
              <p>JD.com opened 2027 campus recruiting for Java backend engineers.</p>
              <p>Agent platform engineering and RAG application roles are also listed.</p>
              <a href="https://campus.jd.com/jobs/java-backend">Apply link</a>
            </div>
          </body>
        </html>
        """
        fetcher = WeChatArticleFetcher(
            client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, text=html, headers={"content-type": "text/html"})
                )
            )
        )

        result = fetcher.fetch(
            "https://mp.weixin.qq.com/s/example",
            ToolCallContext(
                stage="wechat_article_fetch",
                tool_name="WeChatArticleFetcher",
                source_type=JobSourceType.WECHAT_ARTICLE,
                domain="mp.weixin.qq.com",
            ),
        )

        self.assertTrue(result.ok)
        self.assertEqual(ToolSuggestedNextAction.CONTINUE_WORKFLOW, result.suggested_next_action)
        self.assertEqual("2027 Autumn Recruiting Summary", result.artifacts["title"])
        self.assertEqual("DLMU Career", result.artifacts["author"])
        self.assertEqual("2026-08-10", result.artifacts["published_at"])
        self.assertIn("Java backend engineers", result.artifacts["text"])
        self.assertIn("RAG application roles", result.artifacts["text"])
        self.assertEqual(
            ["https://campus.jd.com/jobs/java-backend"],
            result.artifacts["candidate_links"],
        )

    def test_wechat_article_fetcher_returns_mcp_visible_page_when_access_is_restricted(self):
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode, ToolSuggestedNextAction
        from app.domains.jobs.tool_guard import ToolCallContext
        from app.domains.jobs.wechat_fetcher import WeChatArticleFetcher

        html = "<html><body>Please open this link in WeChat client. captcha required.</body></html>"
        fetcher = WeChatArticleFetcher(
            client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=html)))
        )

        result = fetcher.fetch(
            "https://mp.weixin.qq.com/s/restricted",
            ToolCallContext(
                stage="wechat_article_fetch",
                tool_name="WeChatArticleFetcher",
                source_type=JobSourceType.WECHAT_ARTICLE,
                domain="mp.weixin.qq.com",
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(ToolErrorCode.REQUIRES_MCP_VISIBLE_PAGE, result.error_code)
        self.assertEqual(ToolSuggestedNextAction.REQUEST_USER_VISIBLE_PAGE, result.suggested_next_action)
        self.assertTrue(result.retryable)

    def test_wechat_article_fetcher_returns_manual_paste_fallback_for_empty_or_short_article(self):
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode, ToolSuggestedNextAction
        from app.domains.jobs.tool_guard import ToolCallContext
        from app.domains.jobs.wechat_fetcher import WeChatArticleFetcher

        html = """
        <html><body>
          <h1 id="activity-name">Short article</h1>
          <div id="js_content">OK</div>
        </body></html>
        """
        fetcher = WeChatArticleFetcher(
            client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=html))),
            min_text_length=20,
        )

        result = fetcher.fetch(
            "https://mp.weixin.qq.com/s/short",
            ToolCallContext(
                stage="wechat_article_fetch",
                tool_name="WeChatArticleFetcher",
                source_type=JobSourceType.WECHAT_ARTICLE,
                domain="mp.weixin.qq.com",
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(ToolErrorCode.CONTENT_TOO_SHORT, result.error_code)
        self.assertEqual(ToolSuggestedNextAction.REQUEST_MANUAL_PASTE, result.suggested_next_action)
        self.assertEqual("OK", result.artifacts["text"])

    def test_wechat_article_fetcher_runs_tool_guard_before_network_request(self):
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode
        from app.domains.jobs.tool_guard import ToolCallContext
        from app.domains.jobs.wechat_fetcher import WeChatArticleFetcher

        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            return httpx.Response(200, text="<html>should not fetch</html>")

        fetcher = WeChatArticleFetcher(client=httpx.Client(transport=httpx.MockTransport(handler)))
        result = fetcher.fetch(
            "https://mp.weixin.qq.com/s/example",
            ToolCallContext(
                stage="wechat_article_fetch",
                tool_name="WeChatArticleFetcher",
                source_type=JobSourceType.PUBLIC_ARTICLE,
                domain="mp.weixin.qq.com",
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(ToolErrorCode.SOURCE_TYPE_NOT_ALLOWED, result.error_code)
        self.assertEqual(0, calls["count"])


if __name__ == "__main__":
    unittest.main()
