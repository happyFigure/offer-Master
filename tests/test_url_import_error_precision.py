import sys
import unittest
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class RaisingRenderer:
    def __init__(self, exc: Exception):
        self.exc = exc

    def render(self, url: str):
        raise self.exc


class RaisingExtractor:
    def __init__(self, exc: Exception):
        self.exc = exc

    def extract(self, *, html: str, url: str):
        raise self.exc


class UrlImportErrorPrecisionTest(unittest.TestCase):
    def test_tool_result_exposes_machine_readable_error_details(self):
        from app.domains.jobs.schemas import ToolErrorCode, ToolResult, ToolSuggestedNextAction

        result = ToolResult(
            ok=False,
            stage="http_article_fetch",
            tool_name="HTTPArticleFetcher",
            error_code=ToolErrorCode.FETCH_FAILED,
            error_message="HTTP failed",
            suggested_next_action=ToolSuggestedNextAction.RETRY_WITH_NEXT_FETCHER,
            error_details={"category": "http_status", "status_code": 403},
        )

        self.assertEqual("http_status", result.error_details["category"])
        self.assertEqual(403, result.error_details["status_code"])

    def test_guard_policy_rejection_includes_precise_policy_details(self):
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode
        from app.domains.jobs.tool_guard import ToolCallContext, ToolRuntimeGuard

        result = ToolRuntimeGuard().pre_check(
            ToolCallContext(
                stage="http_article_fetch",
                tool_name="HTTPArticleFetcher",
                source_type=JobSourceType.XIAOHONGSHU_NOTE,
                domain="www.xiaohongshu.com",
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(ToolErrorCode.SOURCE_TYPE_NOT_ALLOWED, result.error_code)
        self.assertEqual("source_type_policy", result.error_details["category"])
        self.assertEqual("xiaohongshu_note", result.error_details["source_type"])
        self.assertEqual("HTTPArticleFetcher", result.error_details["tool_name"])
        self.assertIn("MCPVisiblePageFetcher", result.error_details["allowed_tools_for_source"])

    def test_http_status_failure_includes_status_family_final_url_and_preview(self):
        from app.domains.jobs.content_fetcher import HTTPArticleFetcher
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode
        from app.domains.jobs.tool_guard import ToolCallContext

        fetcher = HTTPArticleFetcher(
            client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        403,
                        text="Forbidden by anti bot policy",
                        headers={"content-type": "text/html"},
                        request=request,
                    )
                )
            )
        )

        result = fetcher.fetch(
            "https://career.example.com/private",
            ToolCallContext(
                stage="http_article_fetch",
                tool_name="HTTPArticleFetcher",
                source_type=JobSourceType.OFFICIAL_CAREER_SITE,
                domain="career.example.com",
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(ToolErrorCode.FETCH_FAILED, result.error_code)
        self.assertEqual("http_status", result.error_details["category"])
        self.assertEqual(403, result.error_details["status_code"])
        self.assertEqual("4xx", result.error_details["status_family"])
        self.assertEqual("HTTPStatusError", result.error_details["exception_type"])
        self.assertEqual("https://career.example.com/private", result.error_details["final_url"])
        self.assertIn("anti bot", result.error_details["response_preview"])

    def test_timeout_failure_includes_timeout_category_and_exception_type(self):
        from app.domains.jobs.content_fetcher import HTTPArticleFetcher
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode
        from app.domains.jobs.tool_guard import ToolCallContext

        fetcher = HTTPArticleFetcher(
            client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("read timed out"))
                )
            )
        )

        result = fetcher.fetch(
            "https://example.com/jobs",
            ToolCallContext(
                stage="http_article_fetch",
                tool_name="HTTPArticleFetcher",
                source_type=JobSourceType.PUBLIC_ARTICLE,
                domain="example.com",
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(ToolErrorCode.FETCH_TIMEOUT, result.error_code)
        self.assertEqual("network_timeout", result.error_details["category"])
        self.assertEqual("ReadTimeout", result.error_details["exception_type"])

    def test_wechat_access_restriction_includes_detected_marker_and_preview(self):
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode
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
        self.assertEqual("access_restricted", result.error_details["category"])
        self.assertEqual("please open this link in wechat client", result.error_details["detected_marker"])
        self.assertIn("captcha", result.error_details["text_preview"])

    def test_playwright_exception_includes_renderer_exception_type(self):
        from app.domains.jobs.js_render_fetcher import PlaywrightFetcher
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode
        from app.domains.jobs.tool_guard import ToolCallContext

        result = PlaywrightFetcher(renderer=RaisingRenderer(RuntimeError("browser crashed"))).fetch(
            "https://career.example.com/jobs",
            ToolCallContext(
                stage="js_render_fetch",
                tool_name="PlaywrightFetcher",
                source_type=JobSourceType.OFFICIAL_CAREER_SITE,
                domain="career.example.com",
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(ToolErrorCode.FETCH_FAILED, result.error_code)
        self.assertEqual("renderer_exception", result.error_details["category"])
        self.assertEqual("RuntimeError", result.error_details["exception_type"])

    def test_crawl4ai_extraction_exception_includes_extraction_exception_type(self):
        from app.domains.jobs.js_render_fetcher import Crawl4AIFetcher
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode
        from app.domains.jobs.tool_guard import ToolCallContext

        fetcher = Crawl4AIFetcher(
            client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text="<html></html>"))),
            extractor=RaisingExtractor(ValueError("parser failed")),
        )

        result = fetcher.fetch(
            "https://example.com/jobs",
            ToolCallContext(
                stage="crawl4ai_extract",
                tool_name="Crawl4AIFetcher",
                source_type=JobSourceType.PUBLIC_ARTICLE,
                domain="example.com",
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(ToolErrorCode.CONTENT_EXTRACTION_FAILED, result.error_code)
        self.assertEqual("content_extraction_exception", result.error_details["category"])
        self.assertEqual("ValueError", result.error_details["exception_type"])


if __name__ == "__main__":
    unittest.main()
