import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class MemoryCandidateExtractorTest(unittest.TestCase):
    def test_extracts_explicit_application_confirmation_boundary_from_user_message(self):
        from app.agent_runtime.memory.memory_candidate_extractor import extract_memory_candidates
        from app.domains.agent_memory.models import (
            AgentLearningCandidateRiskLevel,
            AgentLearningCandidateLessonType,
        )
        from app.domains.conversations.models import AgentMessage, AgentMessageKind, AgentMessageRole

        message = AgentMessage(
            id="message-1",
            session_id="session-1",
            role=AgentMessageRole.USER,
            message_kind=AgentMessageKind.USER_TEXT,
            content_text="投递前一定要让我确认，不要自动提交。",
            visible_content_text="投递前一定要让我确认，不要自动提交。",
        )

        drafts = extract_memory_candidates(messages=[message], tool_logs=[])

        self.assertEqual(1, len(drafts))
        draft = drafts[0]
        self.assertEqual("user_preference", draft.memory_type)
        self.assertEqual("application_submission", draft.scope)
        self.assertGreaterEqual(draft.importance, 90)
        self.assertEqual(AgentLearningCandidateRiskLevel.HIGH, draft.risk_level)
        self.assertEqual(AgentLearningCandidateLessonType.USER_PREFERENCE, draft.lesson_type)
        self.assertEqual(("message-1",), draft.evidence_ids)
        self.assertIn("确认", draft.content)

    def test_extracts_data_integrity_preference_from_user_message(self):
        from app.agent_runtime.memory.memory_candidate_extractor import extract_memory_candidates
        from app.domains.conversations.models import AgentMessage, AgentMessageKind, AgentMessageRole

        message = AgentMessage(
            id="message-2",
            session_id="session-1",
            role=AgentMessageRole.USER,
            message_kind=AgentMessageKind.USER_TEXT,
            content_text="判断不了的企业性质就留空，不要编造。",
            visible_content_text="判断不了的企业性质就留空，不要编造。",
        )

        drafts = extract_memory_candidates(messages=[message], tool_logs=[])

        self.assertEqual(1, len(drafts))
        self.assertEqual("data_integrity", drafts[0].scope)
        self.assertEqual(("message-2",), drafts[0].evidence_ids)
        self.assertIn("留空", drafts[0].content)

    def test_extracts_reusable_recovery_lesson_from_failed_and_successful_tool_logs(self):
        from app.agent_runtime.memory.memory_candidate_extractor import extract_memory_candidates
        from app.domains.automation.models import ToolCallLog, ToolCallStatus

        failed_log = ToolCallLog(
            id="tool-log-failed",
            workflow_run_id="workflow-1",
            tool_name="WeChatArticleFetcher",
            tool_group="content_fetcher",
            status=ToolCallStatus.FAILED,
            input_payload={"url": "https://example.com/article"},
            error="EXTRACTION_EMPTY",
        )
        recovered_log = ToolCallLog(
            id="tool-log-success",
            workflow_run_id="workflow-1",
            tool_name="WeChatArticleFetcher",
            tool_group="content_fetcher",
            status=ToolCallStatus.SUCCEEDED,
            input_payload={"url": "https://example.com/article"},
            output_payload={
                "recovery_path": "ask for visible article body and extract recruiting signals",
                "extracted_count": 2,
                "verified": True,
            },
        )

        drafts = extract_memory_candidates(messages=[], tool_logs=[failed_log, recovered_log])

        self.assertEqual(1, len(drafts))
        draft = drafts[0]
        self.assertEqual("tool_recovery", draft.memory_type)
        self.assertEqual("content_fetcher", draft.scope)
        self.assertIn("visible article body", draft.content)
        self.assertEqual(
            ("tool-log-failed", "tool-log-success"),
            draft.evidence_ids,
        )

    def test_ignores_transient_timeout_without_recovery(self):
        from app.agent_runtime.memory.memory_candidate_extractor import extract_memory_candidates
        from app.domains.automation.models import ToolCallLog, ToolCallStatus

        failed_log = ToolCallLog(
            id="tool-log-timeout",
            workflow_run_id="workflow-1",
            tool_name="HttpFetcher",
            tool_group="content_fetcher",
            status=ToolCallStatus.FAILED,
            input_payload={"url": "https://example.com"},
            error="httpx.ReadTimeout: transient timeout",
        )

        self.assertEqual([], extract_memory_candidates(messages=[], tool_logs=[failed_log]))

    def test_redacts_sensitive_values_from_recovery_metadata(self):
        from app.agent_runtime.memory.memory_candidate_extractor import extract_memory_candidates
        from app.domains.automation.models import ToolCallLog, ToolCallStatus

        failed_log = ToolCallLog(
            id="tool-log-failed",
            workflow_run_id="workflow-1",
            tool_name="XhsFetcher",
            tool_group="content_fetcher",
            status=ToolCallStatus.FAILED,
            input_payload={"cookie": "secret-cookie", "url": "https://example.com"},
            error="LOGIN_REQUIRED",
        )
        recovered_log = ToolCallLog(
            id="tool-log-success",
            workflow_run_id="workflow-1",
            tool_name="XhsFetcher",
            tool_group="content_fetcher",
            status=ToolCallStatus.SUCCEEDED,
            input_payload={"cookie": "secret-cookie", "url": "https://example.com"},
            output_payload={"recovery_path": "check local MCP health before searching"},
        )

        drafts = extract_memory_candidates(messages=[], tool_logs=[failed_log, recovered_log])

        self.assertEqual(1, len(drafts))
        self.assertNotIn("secret-cookie", repr(drafts[0].metadata))
        self.assertEqual("[REDACTED]", drafts[0].metadata["failed_input"]["cookie"])


if __name__ == "__main__":
    unittest.main()
