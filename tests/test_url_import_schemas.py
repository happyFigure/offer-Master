import sys
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class UrlImportSchemasTest(unittest.TestCase):
    def test_import_url_request_normalizes_basic_inputs_and_rejects_non_http_urls(self):
        from app.domains.jobs.models import JobSourceTrustLevel, JobSourceType
        from app.domains.jobs.schemas import ImportUrlRequest

        request = ImportUrlRequest(
            url="  https://mp.weixin.qq.com/s/example?utm_source=xhs  ",
            source_id="source-1",
            source_hint="wechat_article",
            trust_level="medium_high",
        )

        self.assertEqual("https://mp.weixin.qq.com/s/example?utm_source=xhs", request.url)
        self.assertEqual("source-1", request.source_id)
        self.assertEqual(JobSourceType.WECHAT_ARTICLE, request.source_hint)
        self.assertEqual(JobSourceTrustLevel.MEDIUM_HIGH, request.trust_level)
        self.assertFalse(request.force_refresh)

        with self.assertRaises(ValidationError):
            ImportUrlRequest(url="ftp://example.com/jobs")

    def test_url_import_accepted_response_exposes_run_state_and_domain_health(self):
        from app.domains.jobs.models import DomainHealthState, UrlImportRunStatus
        from app.domains.jobs.schemas import ImportUrlAcceptedResponse

        response = ImportUrlAcceptedResponse(
            run_id="run-1",
            status=UrlImportRunStatus.RUNNING,
            current_stage="created",
            domain_health_state=DomainHealthState.CLOSED,
            message="URL import accepted",
        )

        self.assertEqual(
            {
                "run_id": "run-1",
                "status": "running",
                "current_stage": "created",
                "domain_health_state": "closed",
                "message": "URL import accepted",
            },
            response.model_dump(mode="json"),
        )

    def test_run_and_domain_health_read_models_serialize_runtime_state(self):
        from app.domains.jobs.models import DomainHealthState, JobSourceType, UrlImportRunStatus
        from app.domains.jobs.schemas import DomainHealthRead, UrlImportRunRead

        now = datetime(2026, 8, 12, 10, 30, tzinfo=UTC)
        run_read = UrlImportRunRead.model_validate(
            SimpleNamespace(
                id="run-1",
                workflow_run_id="workflow-1",
                source_id="source-1",
                input_url="https://mp.weixin.qq.com/s/example?utm_source=xhs",
                normalized_url="https://mp.weixin.qq.com/s/example",
                normalized_url_hash="abc123",
                source_type=JobSourceType.WECHAT_ARTICLE,
                domain="mp.weixin.qq.com",
                fetch_layer="wechat_article",
                status=UrlImportRunStatus.FAILED_RECOVERABLE,
                current_stage="wechat_article_fetch",
                attempt_count=3,
                tool_call_count=4,
                llm_call_count=1,
                error_code="FETCH_TIMEOUT",
                error_message="Request timed out",
                next_action="retry_after_cooldown",
                raw_job_lead_id="raw-1",
                extracted_count=0,
                duplicate_of_run_id=None,
                run_metadata={"source_hint": "wechat_article"},
                started_at=now,
                updated_at=now,
                finished_at=None,
            )
        )
        health_read = DomainHealthRead.model_validate(
            SimpleNamespace(
                id="health-1",
                domain="mp.weixin.qq.com",
                tool_name="WeChatArticleFetcher",
                state=DomainHealthState.OPEN,
                failure_count=5,
                success_count=18,
                last_error_code="HTTP_429",
                last_error_message="too many requests",
                opened_at=now,
                cooldown_until=now + timedelta(minutes=30),
                half_open_probe_count=0,
                created_at=now,
                updated_at=now,
            )
        )

        run_json = run_read.model_dump(mode="json")
        health_json = health_read.model_dump(mode="json")

        self.assertEqual("failed_recoverable", run_json["status"])
        self.assertEqual("wechat_article", run_json["source_type"])
        self.assertEqual("FETCH_TIMEOUT", run_json["error_code"])
        self.assertEqual("retry_after_cooldown", run_json["next_action"])
        self.assertEqual("open", health_json["state"])
        self.assertEqual("HTTP_429", health_json["last_error_code"])

    def test_tool_result_structures_errors_with_code_and_next_action(self):
        from app.domains.jobs.schemas import ToolErrorCode, ToolResult, ToolSuggestedNextAction

        result = ToolResult(
            ok=False,
            stage="http_article_fetch",
            tool_name="HTTPArticleFetcher",
            error_code=ToolErrorCode.TOOL_CIRCUIT_OPEN,
            error_message="Circuit is open for this domain",
            retryable=True,
            suggested_next_action=ToolSuggestedNextAction.WAIT_FOR_COOLDOWN,
            cost={"tool_calls": 1, "llm_calls": 0},
            artifacts={"domain": "example.com"},
        )

        self.assertEqual(
            {
                "ok": False,
                "stage": "http_article_fetch",
                "tool_name": "HTTPArticleFetcher",
                "error_code": "TOOL_CIRCUIT_OPEN",
                "error_message": "Circuit is open for this domain",
                "retryable": True,
                "suggested_next_action": "wait_for_cooldown",
                "error_details": {},
                "cost": {"tool_calls": 1, "llm_calls": 0},
                "artifacts": {"domain": "example.com"},
            },
            result.model_dump(mode="json"),
        )
        self.assertIn("TOOL_BUDGET_EXCEEDED", {item.value for item in ToolErrorCode})
        self.assertIn("retry_with_next_fetcher", {item.value for item in ToolSuggestedNextAction})


if __name__ == "__main__":
    unittest.main()
