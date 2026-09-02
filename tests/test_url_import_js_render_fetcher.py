import sys
import unittest
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class FakeRenderer:
    def __init__(self, page):
        self.page = page
        self.calls = 0

    def render(self, url: str):
        self.calls += 1
        return self.page


class FakeCrawl4AIExtractor:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def extract(self, *, html: str, url: str):
        self.calls += 1
        return self.content


class UrlImportJsRenderFetcherTest(unittest.TestCase):
    def test_playwright_fetcher_returns_visible_text_from_rendered_public_page(self):
        from app.domains.jobs.js_render_fetcher import PlaywrightFetcher, RenderedPage
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolSuggestedNextAction
        from app.domains.jobs.tool_guard import ToolCallContext

        page = RenderedPage(
            final_url="https://campus.jd.com/jobs/agent-engineer",
            status_code=200,
            title="JD 2027 Agent Platform Engineer",
            html="<html><body><div id='root'>rendered by js</div></body></html>",
            visible_text=(
                "JD.com 2027 campus recruiting has opened Agent platform engineer roles. "
                "The role focuses on LLM workflow orchestration, RAG applications, and backend services."
            ),
            candidate_links=["https://campus.jd.com/apply/agent-engineer"],
        )
        renderer = FakeRenderer(page)
        fetcher = PlaywrightFetcher(renderer=renderer, min_text_length=80)

        result = fetcher.fetch(
            "https://campus.jd.com/jobs/agent-engineer",
            ToolCallContext(
                stage="js_render_fetch",
                tool_name="PlaywrightFetcher",
                source_type=JobSourceType.OFFICIAL_CAREER_SITE,
                domain="campus.jd.com",
            ),
        )

        self.assertTrue(result.ok)
        self.assertEqual(1, renderer.calls)
        self.assertEqual(ToolSuggestedNextAction.CONTINUE_WORKFLOW, result.suggested_next_action)
        self.assertEqual("JD 2027 Agent Platform Engineer", result.artifacts["title"])
        self.assertIn("LLM workflow orchestration", result.artifacts["text"])
        self.assertEqual("playwright_visible_text", result.artifacts["extraction_method"])
        self.assertEqual(
            ["https://campus.jd.com/apply/agent-engineer"],
            result.artifacts["candidate_links"],
        )

    def test_playwright_fetcher_routes_login_or_captcha_to_user_visible_boundary(self):
        from app.domains.jobs.js_render_fetcher import PlaywrightFetcher, RenderedPage
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode, ToolSuggestedNextAction
        from app.domains.jobs.tool_guard import ToolCallContext

        page = RenderedPage(
            final_url="https://career.example.com/jobs/private",
            status_code=200,
            title="Login required",
            html="<html><body>captcha required</body></html>",
            visible_text="Login required. Please verify captcha before continuing.",
            candidate_links=[],
        )

        result = PlaywrightFetcher(renderer=FakeRenderer(page), min_text_length=20).fetch(
            "https://career.example.com/jobs/private",
            ToolCallContext(
                stage="js_render_fetch",
                tool_name="PlaywrightFetcher",
                source_type=JobSourceType.OFFICIAL_CAREER_SITE,
                domain="career.example.com",
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(ToolErrorCode.REQUIRES_MCP_VISIBLE_PAGE, result.error_code)
        self.assertEqual(ToolSuggestedNextAction.REQUEST_USER_VISIBLE_PAGE, result.suggested_next_action)
        self.assertTrue(result.retryable)

    def test_playwright_fetcher_runs_tool_guard_before_renderer(self):
        from app.domains.jobs.js_render_fetcher import PlaywrightFetcher, RenderedPage
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode
        from app.domains.jobs.tool_guard import ToolCallContext

        renderer = FakeRenderer(
            RenderedPage(
                final_url="https://www.xiaohongshu.com/discovery/item/demo",
                status_code=200,
                title="Should not render",
                html="<html></html>",
                visible_text="Should not render",
                candidate_links=[],
            )
        )

        result = PlaywrightFetcher(renderer=renderer).fetch(
            "https://www.xiaohongshu.com/discovery/item/demo",
            ToolCallContext(
                stage="js_render_fetch",
                tool_name="PlaywrightFetcher",
                source_type=JobSourceType.XIAOHONGSHU_NOTE,
                domain="www.xiaohongshu.com",
            ),
        )

        self.assertFalse(result.ok)
        self.assertEqual(ToolErrorCode.SOURCE_TYPE_NOT_ALLOWED, result.error_code)
        self.assertEqual(0, renderer.calls)

    def test_playwright_runtime_paths_default_to_non_c_project_directories(self):
        from app.domains.jobs.js_render_fetcher import build_playwright_runtime_paths

        paths = build_playwright_runtime_paths(PROJECT_ROOT)

        self.assertEqual(PROJECT_ROOT / ".external" / "ms-playwright", paths.browsers_path)
        self.assertEqual(PROJECT_ROOT / "runtime" / "playwright" / "user-data", paths.user_data_dir)
        self.assertEqual(PROJECT_ROOT / "runtime" / "temp", paths.temp_dir)
        for path in (paths.browsers_path, paths.user_data_dir, paths.temp_dir):
            self.assertFalse(str(path).lower().startswith("c:"))

    def test_crawl4ai_fetcher_enhances_html_to_markdown_and_text(self):
        from app.domains.jobs.js_render_fetcher import Crawl4AIExtractedContent, Crawl4AIFetcher
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolSuggestedNextAction
        from app.domains.jobs.tool_guard import ToolCallContext

        html = "<html><main><h1>Alibaba Cloud 2027 Java Backend</h1></main></html>"
        content = Crawl4AIExtractedContent(
            title="Alibaba Cloud 2027 Java Backend",
            markdown="# Alibaba Cloud 2027 Java Backend\nApply for Java backend and Agent roles.",
            text=(
                "Alibaba Cloud 2027 campus recruiting includes Java backend, Agent application "
                "engineering, and RAG platform roles for computer science students."
            ),
            candidate_links=["https://talent.alibaba.com/campus/apply/java-backend"],
            extraction_method="crawl4ai_markdown",
        )
        fetcher = Crawl4AIFetcher(
            client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=html))),
            extractor=FakeCrawl4AIExtractor(content),
            min_text_length=80,
        )

        result = fetcher.fetch(
            "https://talent.alibaba.com/campus/jobs/java-backend",
            ToolCallContext(
                stage="crawl4ai_extract",
                tool_name="Crawl4AIFetcher",
                source_type=JobSourceType.OFFICIAL_CAREER_SITE,
                domain="talent.alibaba.com",
            ),
        )

        self.assertTrue(result.ok)
        self.assertEqual(ToolSuggestedNextAction.CONTINUE_WORKFLOW, result.suggested_next_action)
        self.assertIn("Agent application", result.artifacts["text"])
        self.assertIn("# Alibaba Cloud", result.artifacts["markdown"])
        self.assertEqual("crawl4ai_markdown", result.artifacts["extraction_method"])

    def test_crawl4ai_fetcher_returns_content_too_short_when_extraction_is_empty(self):
        from app.domains.jobs.js_render_fetcher import Crawl4AIExtractedContent, Crawl4AIFetcher
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode, ToolSuggestedNextAction
        from app.domains.jobs.tool_guard import ToolCallContext

        fetcher = Crawl4AIFetcher(
            client=httpx.Client(
                transport=httpx.MockTransport(lambda request: httpx.Response(200, text="<html>OK</html>"))
            ),
            extractor=FakeCrawl4AIExtractor(
                Crawl4AIExtractedContent(
                    title="OK",
                    markdown="OK",
                    text="OK",
                    candidate_links=[],
                    extraction_method="crawl4ai_markdown",
                )
            ),
            min_text_length=20,
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
        self.assertEqual(ToolErrorCode.CONTENT_TOO_SHORT, result.error_code)
        self.assertEqual(ToolSuggestedNextAction.REQUEST_MANUAL_PASTE, result.suggested_next_action)

    def test_requires_javascript_error_code_is_available_for_workflow_routing(self):
        from app.domains.jobs.schemas import ToolErrorCode

        self.assertEqual("REQUIRES_JAVASCRIPT", ToolErrorCode.REQUIRES_JAVASCRIPT.value)


if __name__ == "__main__":
    unittest.main()
