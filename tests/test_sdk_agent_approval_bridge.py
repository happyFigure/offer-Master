import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class SdkAgentApprovalBridgeTest(unittest.TestCase):
    def test_builds_wait_confirmation_payload_from_sdk_needs_approval_result(self) -> None:
        from app.agent_runtime.agent_as_tool import StandardAgentResult
        from app.agent_runtime.sdk_agents.approval_bridge import sdk_agent_approval_payload_from_result
        from app.agent_runtime.sdk_agents.schemas import SdkAgentResultEnvelope, SdkToolApprovalRequest, SdkToolTraceSummary

        sdk_result = SdkAgentResultEnvelope(
            task_id="run-1:tool-1",
            capability_id="filesystem.read_file",
            subagent_name="FileAnalysisAgent",
            status="needs_approval",
            summary="需要确认后才能删除临时文件。",
            approval_request=SdkToolApprovalRequest(
                approval_type="tool_call",
                tool_name="filesystem.delete_path",
                tool_input={"path": "data/cache/tmp.json", "force": True},
                reason="删除文件属于高风险动作。",
                suggested_user_message="确认删除 data/cache/tmp.json 吗？",
            ),
            trace_summary=SdkToolTraceSummary(tool_call_count=1),
            raw_trace_ref="trace-sdk-1",
        ).to_standard_agent_result()

        payload = sdk_agent_approval_payload_from_result(
            sdk_result,
            outer_capability="filesystem.read_file",
            outer_tool_input={"path": "data/cache"},
            executor_id="openai-sdk-agent",
        )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertTrue(payload["sdk_agent_approval"])
        self.assertEqual("filesystem.delete_path", payload["requested_tool_name"])
        self.assertEqual({"path": "data/cache/tmp.json", "force": True}, payload["tool_input"])
        self.assertEqual("filesystem.read_file", payload["outer_capability"])
        self.assertEqual({"path": "data/cache"}, payload["outer_tool_input"])
        self.assertEqual("openai-sdk-agent", payload["executor_id"])
        self.assertEqual("trace-sdk-1", payload["raw_trace_ref"])
        self.assertEqual("SDK_AGENT_TOOL_APPROVAL_REQUIRED", payload["guard_result"]["error_code"])
        self.assertEqual("wait_confirmation", payload["guard_result"]["next_action"])

    def test_returns_none_for_normal_agent_result(self) -> None:
        from app.agent_runtime.agent_as_tool import StandardAgentResult
        from app.agent_runtime.sdk_agents.approval_bridge import sdk_agent_approval_payload_from_result

        payload = sdk_agent_approval_payload_from_result(
            StandardAgentResult(status="succeeded", summary="ok"),
            outer_capability="external.web_search",
            outer_tool_input={"query": "腾讯校招"},
            executor_id="openai-sdk-agent",
        )

        self.assertIsNone(payload)


if __name__ == "__main__":
    unittest.main()
