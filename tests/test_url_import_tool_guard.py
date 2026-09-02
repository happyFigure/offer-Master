import sys
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class UrlImportToolGuardTest(unittest.TestCase):
    def test_pre_check_allows_policy_permitted_tool(self):
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.tool_guard import ToolCallContext, ToolRuntimeGuard

        guard = ToolRuntimeGuard()
        result = guard.pre_check(
            ToolCallContext(
                stage="http_article_fetch",
                tool_name="HTTPArticleFetcher",
                source_type=JobSourceType.PUBLIC_ARTICLE,
                domain="example.com",
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual("continue_workflow", result.suggested_next_action)

    def test_pre_check_rejects_tool_not_in_allowlist(self):
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode, ToolSuggestedNextAction
        from app.domains.jobs.tool_guard import ToolCallContext, ToolRuntimeGuard

        result = ToolRuntimeGuard().pre_check(
            ToolCallContext(
                stage="http_article_fetch",
                tool_name="ShellCommandRunner",
                source_type=JobSourceType.PUBLIC_ARTICLE,
                domain="example.com",
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(ToolErrorCode.TOOL_NOT_ALLOWED, result.error_code)
        self.assertEqual(ToolSuggestedNextAction.STOP_TERMINAL_FAILURE, result.suggested_next_action)

    def test_pre_check_rejects_source_type_disallowed_tool(self):
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode, ToolSuggestedNextAction
        from app.domains.jobs.tool_guard import ToolCallContext, ToolRuntimeGuard

        result = ToolRuntimeGuard().pre_check(
            ToolCallContext(
                stage="http_article_fetch",
                tool_name="HTTPArticleFetcher",
                source_type=JobSourceType.XIAOHONGSHU_NOTE,
                domain="www.xiaohongshu.com",
                user_confirmed=False,
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(ToolErrorCode.SOURCE_TYPE_NOT_ALLOWED, result.error_code)
        self.assertEqual(
            ToolSuggestedNextAction.REQUEST_USER_VISIBLE_PAGE,
            result.suggested_next_action,
        )

    def test_pre_check_enforces_tool_llm_fetch_time_and_mcp_budgets(self):
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode
        from app.domains.jobs.tool_guard import ToolCallContext, ToolRuntimeGuard

        now = datetime(2026, 8, 12, 10, 0, tzinfo=UTC).replace(tzinfo=None)
        guard = ToolRuntimeGuard()

        cases = [
            (
                ToolCallContext(
                    stage="http_article_fetch",
                    tool_name="HTTPArticleFetcher",
                    source_type=JobSourceType.PUBLIC_ARTICLE,
                    domain="example.com",
                    tool_call_count=15,
                    now=now,
                ),
                ToolErrorCode.TOOL_BUDGET_EXCEEDED,
            ),
            (
                ToolCallContext(
                    stage="llm_extract",
                    tool_name="BailianJobLeadExtractor",
                    source_type=JobSourceType.PUBLIC_ARTICLE,
                    domain="example.com",
                    llm_call_count=3,
                    now=now,
                ),
                ToolErrorCode.LLM_BUDGET_EXCEEDED,
            ),
            (
                ToolCallContext(
                    stage="http_article_fetch",
                    tool_name="HTTPArticleFetcher",
                    source_type=JobSourceType.PUBLIC_ARTICLE,
                    domain="example.com",
                    fetch_attempts_for_stage=3,
                    now=now,
                ),
                ToolErrorCode.FETCH_ATTEMPTS_EXCEEDED,
            ),
            (
                ToolCallContext(
                    stage="http_article_fetch",
                    tool_name="HTTPArticleFetcher",
                    source_type=JobSourceType.PUBLIC_ARTICLE,
                    domain="example.com",
                    run_started_at=now - timedelta(seconds=181),
                    now=now,
                ),
                ToolErrorCode.TIME_BUDGET_EXCEEDED,
            ),
            (
                ToolCallContext(
                    stage="mcp_visible_page",
                    tool_name="MCPVisiblePageFetcher",
                    source_type=JobSourceType.XIAOHONGSHU_NOTE,
                    domain="www.xiaohongshu.com",
                    mcp_request_count=1,
                    user_confirmed=True,
                    now=now,
                ),
                ToolErrorCode.TOOL_BUDGET_EXCEEDED,
            ),
        ]

        for context, error_code in cases:
            with self.subTest(error_code=error_code):
                result = guard.pre_check(context)
                self.assertFalse(result.ok)
                self.assertEqual(error_code, result.error_code)

    def test_pre_check_requires_user_confirmation_for_mcp_visible_page(self):
        from app.domains.jobs.models import JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode, ToolSuggestedNextAction
        from app.domains.jobs.tool_guard import ToolCallContext, ToolRuntimeGuard

        result = ToolRuntimeGuard().pre_check(
            ToolCallContext(
                stage="mcp_visible_page",
                tool_name="MCPVisiblePageFetcher",
                source_type=JobSourceType.XIAOHONGSHU_NOTE,
                domain="www.xiaohongshu.com",
                user_confirmed=False,
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(ToolErrorCode.MCP_USER_CONFIRMATION_REQUIRED, result.error_code)
        self.assertEqual(
            ToolSuggestedNextAction.REQUEST_USER_VISIBLE_PAGE,
            result.suggested_next_action,
        )

    def test_pre_check_blocks_open_circuit_and_allows_half_open_probe_after_cooldown(self):
        from app.domains.jobs.models import DomainHealth, DomainHealthState, JobSourceType
        from app.domains.jobs.schemas import ToolErrorCode, ToolSuggestedNextAction
        from app.domains.jobs.tool_guard import ToolCallContext, ToolRuntimeGuard

        now = datetime(2026, 8, 12, 10, 0, tzinfo=UTC).replace(tzinfo=None)
        health = DomainHealth(
            domain="example.com",
            tool_name="HTTPArticleFetcher",
            state=DomainHealthState.OPEN,
            failure_count=3,
            cooldown_until=now + timedelta(minutes=10),
        )
        context = ToolCallContext(
            stage="http_article_fetch",
            tool_name="HTTPArticleFetcher",
            source_type=JobSourceType.PUBLIC_ARTICLE,
            domain="example.com",
            now=now,
        )

        blocked = ToolRuntimeGuard().pre_check(context, domain_health=health)
        self.assertFalse(blocked.ok)
        self.assertEqual(ToolErrorCode.TOOL_CIRCUIT_OPEN, blocked.error_code)
        self.assertEqual(ToolSuggestedNextAction.WAIT_FOR_COOLDOWN, blocked.suggested_next_action)

        health.cooldown_until = now - timedelta(seconds=1)
        allowed = ToolRuntimeGuard().pre_check(context, domain_health=health)
        self.assertTrue(allowed.ok)
        self.assertEqual(DomainHealthState.HALF_OPEN, health.state)
        self.assertEqual(1, health.half_open_probe_count)

    def test_post_record_opens_circuit_after_failures_and_closes_after_success(self):
        from app.domains.jobs.models import DomainHealth, DomainHealthState
        from app.domains.jobs.schemas import ToolErrorCode, ToolResult, ToolSuggestedNextAction
        from app.domains.jobs.tool_guard import ToolRuntimeGuard

        now = datetime(2026, 8, 12, 10, 0, tzinfo=UTC).replace(tzinfo=None)
        guard = ToolRuntimeGuard()
        health = DomainHealth(domain="example.com", tool_name="HTTPArticleFetcher")
        failed_result = ToolResult(
            ok=False,
            stage="http_article_fetch",
            tool_name="HTTPArticleFetcher",
            error_code=ToolErrorCode.FETCH_TIMEOUT,
            error_message="timed out",
            retryable=True,
            suggested_next_action=ToolSuggestedNextAction.RETRY_SAME_STAGE,
        )

        guard.post_record(failed_result, health, now=now)
        guard.post_record(failed_result, health, now=now + timedelta(seconds=1))
        guard.post_record(failed_result, health, now=now + timedelta(seconds=2))

        self.assertEqual(DomainHealthState.OPEN, health.state)
        self.assertEqual(3, health.failure_count)
        self.assertEqual("FETCH_TIMEOUT", health.last_error_code)
        self.assertIsNotNone(health.cooldown_until)

        success_result = ToolResult(
            ok=True,
            stage="http_article_fetch",
            tool_name="HTTPArticleFetcher",
            suggested_next_action=ToolSuggestedNextAction.CONTINUE_WORKFLOW,
        )
        guard.post_record(success_result, health, now=now + timedelta(minutes=31))

        self.assertEqual(DomainHealthState.CLOSED, health.state)
        self.assertEqual(0, health.failure_count)
        self.assertEqual(1, health.success_count)
        self.assertIsNone(health.last_error_code)
        self.assertIsNone(health.cooldown_until)


if __name__ == "__main__":
    unittest.main()
