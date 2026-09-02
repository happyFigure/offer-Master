import sys
import unittest
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class UrlImportContentFetcherTest(unittest.TestCase):
    def test_http_article_fetcher_extracts_title_text_and_candidate_links_from_public_html(self):
        from app.domains.jobs.content_fetcher import ContentFetcher, HTTPArticleFetcher
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolSuggestedNextAction
        from app.domains.jobs.tool_guard import ToolCallContext

        html = """
        <html>
          <head><title>Fallback title</title><script>window.bad = true</script></head>
          <body>
            <nav>Navigation should not be kept</nav>
            <article>
              <h1>2027 Campus Recruiting Java Backend</h1>
              <p>JD.com 2027 campus recruiting opens Java backend engineer roles in Beijing.</p>
              <p>Students can apply through the official campus portal before September.</p>
              <a href="/apply/java-backend">Apply online</a>
            </article>
            <footer>Footer should not be kept</footer>
          </body>
        </html>
        """
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text=html, headers={"content-type": "text/html"})
        )
        fetcher = HTTPArticleFetcher(client=httpx.Client(transport=transport))

        self.assertIsInstance(fetcher, ContentFetcher)
        result = fetcher.fetch(
            "https://campus.jd.com/jobs/java-backend",
            ToolCallContext(
                stage="http_article_fetch",
                tool_name="HTTPArticleFetcher",
                source_type=JobSourceType.OFFICIAL_CAREER_SITE,
                domain="campus.jd.com",
            ),
        )

        self.assertTrue(result.ok)
        self.assertEqual(ToolSuggestedNextAction.CONTINUE_WORKFLOW, result.suggested_next_action)
        self.assertEqual("2027 Campus Recruiting Java Backend", result.artifacts["title"])
        self.assertIn("Java backend engineer roles", result.artifacts["text"])
        self.assertNotIn("Navigation should not be kept", result.artifacts["text"])
        self.assertNotIn("Footer should not be kept", result.artifacts["text"])
        self.assertGreater(result.artifacts["content_length"], 80)
        self.assertEqual(200, result.artifacts["status_code"])
        self.assertEqual(
            ["https://campus.jd.com/apply/java-backend"],
            result.artifacts["candidate_links"],
        )

    def test_http_article_fetcher_returns_content_too_short_without_fake_content(self):
        from app.domains.jobs.content_fetcher import HTTPArticleFetcher
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode, ToolSuggestedNextAction
        from app.domains.jobs.tool_guard import ToolCallContext

        transport = httpx.MockTransport(lambda request: httpx.Response(200, text="<html>OK</html>"))
        fetcher = HTTPArticleFetcher(client=httpx.Client(transport=transport), min_text_length=20)
        result = fetcher.fetch(
            "https://example.edu/job/101",
            ToolCallContext(
                stage="http_article_fetch",
                tool_name="HTTPArticleFetcher",
                source_type=JobSourceType.UNIVERSITY_CAREER_SITE,
                domain="example.edu",
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(ToolErrorCode.CONTENT_TOO_SHORT, result.error_code)
        self.assertEqual(ToolSuggestedNextAction.RETRY_WITH_NEXT_FETCHER, result.suggested_next_action)
        self.assertTrue(result.retryable)
        self.assertEqual("OK", result.artifacts["text"])

    def test_http_article_fetcher_returns_structured_errors_for_http_and_network_failures(self):
        from app.domains.jobs.content_fetcher import HTTPArticleFetcher
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode
        from app.domains.jobs.tool_guard import ToolCallContext

        http_error_fetcher = HTTPArticleFetcher(
            client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(503, text="Service unavailable")
                )
            )
        )
        timeout_fetcher = HTTPArticleFetcher(
            client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("read timed out"))
                )
            )
        )
        context = ToolCallContext(
            stage="http_article_fetch",
            tool_name="HTTPArticleFetcher",
            source_type=JobSourceType.PUBLIC_ARTICLE,
            domain="example.com",
        )

        http_error = http_error_fetcher.fetch("https://example.com/jobs", context)
        timeout = timeout_fetcher.fetch("https://example.com/jobs", context)

        self.assertFalse(http_error.ok)
        self.assertEqual(ToolErrorCode.FETCH_FAILED, http_error.error_code)
        self.assertEqual(503, http_error.artifacts["status_code"])
        self.assertFalse(timeout.ok)
        self.assertEqual(ToolErrorCode.FETCH_TIMEOUT, timeout.error_code)

    def test_http_article_fetcher_runs_tool_guard_before_network_request(self):
        from app.domains.jobs.content_fetcher import HTTPArticleFetcher
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode, ToolSuggestedNextAction
        from app.domains.jobs.tool_guard import ToolCallContext

        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            return httpx.Response(200, text="<html>should not fetch</html>")

        fetcher = HTTPArticleFetcher(client=httpx.Client(transport=httpx.MockTransport(handler)))
        result = fetcher.fetch(
            "https://www.xiaohongshu.com/discovery/item/demo",
            ToolCallContext(
                stage="http_article_fetch",
                tool_name="HTTPArticleFetcher",
                source_type=JobSourceType.XIAOHONGSHU_NOTE,
                domain="www.xiaohongshu.com",
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(ToolErrorCode.SOURCE_TYPE_NOT_ALLOWED, result.error_code)
        self.assertEqual(
            ToolSuggestedNextAction.REQUEST_USER_VISIBLE_PAGE,
            result.suggested_next_action,
        )
        self.assertEqual(0, calls["count"])


if __name__ == "__main__":
    unittest.main()
