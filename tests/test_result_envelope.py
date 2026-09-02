import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class ResultEnvelopeTest(unittest.TestCase):
    def test_builds_web_search_result_envelope_from_mixed_source_shapes(self) -> None:
        from app.agent_runtime.routing.result_envelope import build_result_envelope
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL

        envelope = build_result_envelope(
            capability=EXTERNAL_WEB_SEARCH_TOOL,
            status="succeeded",
            risk_level="low",
            result_payload={
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
                    "executor_name": "claude-sdk-agent",
                    "answer": "联网搜索结果：中科曙光校园招聘官网。",
                    "sources": [
                        {"title": "中科曙光校招", "url": "https://jobs.example.com/sugon"},
                        "https://example.com/source-2",
                    ],
                },
            },
        )

        self.assertIsNotNone(envelope)
        data = envelope.to_dict()
        self.assertEqual("succeeded", data["status"])
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, data["capability"])
        self.assertEqual("claude-sdk-agent", data["executor"])
        self.assertEqual("联网搜索结果：中科曙光校园招聘官网。", data["summary"])
        self.assertEqual(
            [
                {"type": "url", "title": "中科曙光校招", "url": "https://jobs.example.com/sugon"},
                {"type": "url", "title": "source", "url": "https://example.com/source-2"},
            ],
            data["artifacts"],
        )

    def test_builds_web_search_result_envelope_prefers_executor_artifacts(self) -> None:
        from app.agent_runtime.routing.result_envelope import build_result_envelope
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL

        envelope = build_result_envelope(
            capability=EXTERNAL_WEB_SEARCH_TOOL,
            status="succeeded",
            risk_level="low",
            result_payload={
                "tool_name": EXTERNAL_WEB_SEARCH_TOOL,
                "ok": True,
                "result": {
                    "executor_name": "claude-sdk-agent",
                    "answer": "联网搜索结果：中科曙光校园招聘官网。",
                    "sources": [{"url": "https://jobs.example.com/sugon"}],
                    "artifacts": [
                        {
                            "type": "url",
                            "title": "中科曙光校园招聘官网",
                            "url": "https://jobs.example.com/sugon",
                        }
                    ],
                    "observations": ["官网入口可访问"],
                },
            },
        )

        self.assertIsNotNone(envelope)
        data = envelope.to_dict()
        self.assertEqual(["官网入口可访问"], data["observations"])
        self.assertEqual(
            [{"type": "url", "title": "中科曙光校园招聘官网", "url": "https://jobs.example.com/sugon"}],
            data["artifacts"],
        )

    def test_builds_apply_entry_result_envelope_from_external_dispatch_result(self) -> None:
        from app.agent_runtime.routing.result_envelope import build_result_envelope
        from app.agent_runtime.tool_registry import APPLICATION_FIND_APPLY_ENTRY_TOOL

        envelope = build_result_envelope(
            capability=APPLICATION_FIND_APPLY_ENTRY_TOOL,
            status="succeeded",
            risk_level="medium",
            result_payload={
                "tool_name": APPLICATION_FIND_APPLY_ENTRY_TOOL,
                "ok": True,
                "result": {
                    "task_id": "external-task-1",
                    "task_envelope": {
                        "job": {
                            "job_id": "lead-apply-1",
                            "company_name": "Tencent",
                            "title": "Backend Engineer Intern",
                        }
                    },
                    "dispatch": {
                        "ok": True,
                        "status": "succeeded",
                        "executor_name": "codex_or_multica",
                        "result_status": "found_opened",
                        "apply_url": "https://careers.tencent.com/apply/1",
                        "next_action": "wait_user_review",
                    },
                },
            },
        )

        self.assertIsNotNone(envelope)
        data = envelope.to_dict()
        self.assertEqual("succeeded", data["status"])
        self.assertEqual(APPLICATION_FIND_APPLY_ENTRY_TOOL, data["capability"])
        self.assertEqual("codex_or_multica", data["executor"])
        self.assertIn("Tencent - Backend Engineer Intern", data["summary"])
        self.assertTrue(data["requires_user_action"])
        self.assertEqual("medium", data["risk_level"])
        self.assertEqual(
            [{"type": "url", "title": "application_entry", "url": "https://careers.tencent.com/apply/1"}],
            data["artifacts"],
        )


if __name__ == "__main__":
    unittest.main()
