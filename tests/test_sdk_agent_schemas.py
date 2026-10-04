import sys
import unittest
from pathlib import Path

from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class SdkAgentSchemasTest(unittest.TestCase):
    def test_task_envelope_can_be_built_from_agent_task(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.sdk_agents.schemas import SdkAgentTaskEnvelope

        task = AgentTask(
            capability_id="agent.web_research",
            goal="查一下腾讯校招官方入口",
            input_payload={"query": "腾讯 校招 官网", "max_results": 5},
            constraints=["只返回公开来源"],
            expected_output=["官方入口", "证据链接"],
        )
        context = AgentRuntimeContext(
            session_id="session-1",
            run_id="workflow-1",
            task_id="workflow-1:tool-1",
            permission_scope={"source_type": "agent_chat", "user_confirmed": False},
            metadata={"agent_run_id": "agent-run-1"},
        )

        envelope = SdkAgentTaskEnvelope.from_agent_task(
            task,
            context,
            subagent_name="WebResearchAgent",
            allowed_tools=["web.search", "web.fetch"],
            risk_policy={"mutation_tools": "approval_required"},
            max_turns=8,
        )

        self.assertEqual("offer_master.sdk_agent_task.v1", envelope.schema_version)
        self.assertEqual("workflow-1:tool-1", envelope.task_id)
        self.assertEqual("workflow-1", envelope.trace_id)
        self.assertEqual("agent.web_research", envelope.capability_id)
        self.assertEqual("WebResearchAgent", envelope.subagent_name)
        self.assertEqual("查一下腾讯校招官方入口", envelope.goal)
        self.assertEqual({"query": "腾讯 校招 官网", "max_results": 5}, envelope.input_payload)
        self.assertEqual(["web.search", "web.fetch"], envelope.allowed_tools)
        self.assertEqual(["只返回公开来源"], envelope.constraints)
        self.assertEqual(["官方入口", "证据链接"], envelope.expected_output)
        self.assertEqual({"mutation_tools": "approval_required"}, envelope.risk_policy)
        self.assertEqual(8, envelope.max_turns)
        self.assertEqual("agent_chat", envelope.context_refs["source_type"])
        self.assertEqual("agent-run-1", envelope.metadata["agent_run_id"])

    def test_result_envelope_converts_to_standard_agent_result_without_raw_trace_in_observation(self) -> None:
        from app.agent_runtime.sdk_agents.schemas import SdkAgentResultEnvelope, SdkToolTraceSummary

        result = SdkAgentResultEnvelope(
            task_id="workflow-1:tool-1",
            capability_id="agent.web_research",
            subagent_name="WebResearchAgent",
            status="succeeded",
            summary="找到腾讯校招官方入口。",
            evidence=[{"title": "腾讯校园招聘", "url": "https://join.qq.com"}],
            diagnostics={"retry_count": 2, "confidence": 0.91},
            operation_refs=["sdk-run-1"],
            trace_summary=SdkToolTraceSummary(
                tool_call_count=3,
                retry_count=2,
                approval_request_ids=[],
                operation_refs=["sdk-run-1", "trace-1"],
            ),
            raw_trace_ref="trace-1",
        )

        standard = result.to_standard_agent_result()

        self.assertEqual("succeeded", standard.status)
        self.assertEqual("找到腾讯校招官方入口。", standard.summary)
        self.assertIn("retry_count=2", standard.observation)
        self.assertIn("confidence=0.91", standard.observation)
        self.assertNotIn("raw_trace", standard.observation)
        self.assertEqual([{"title": "腾讯校园招聘", "url": "https://join.qq.com"}], standard.evidence)
        self.assertEqual("agent.web_research", standard.raw_result["capability_id"])
        self.assertEqual("trace-1", standard.raw_result["raw_trace_ref"])

    def test_approval_result_requires_approval_request_payload(self) -> None:
        from app.agent_runtime.sdk_agents.schemas import SdkAgentResultEnvelope

        with self.assertRaises(ValidationError):
            SdkAgentResultEnvelope(
                task_id="workflow-1:tool-1",
                capability_id="agent.file_analysis",
                subagent_name="FileAnalysisAgent",
                status="needs_approval",
                summary="需要确认删除文件。",
            )

    def test_tool_approval_request_marks_mutation_as_user_action(self) -> None:
        from app.agent_runtime.sdk_agents.schemas import SdkAgentResultEnvelope, SdkToolApprovalRequest

        approval = SdkToolApprovalRequest(
            approval_type="tool_call",
            tool_name="filesystem.delete_path",
            tool_input={"path": "data/cache/tmp.json"},
            reason="缓存文件可重新生成，但删除属于高风险动作。",
            risk_level="high",
            suggested_user_message="是否确认删除 data/cache/tmp.json？",
        )
        result = SdkAgentResultEnvelope(
            task_id="workflow-1:tool-1",
            capability_id="agent.file_analysis",
            subagent_name="FileAnalysisAgent",
            status="needs_approval",
            summary="等待用户确认删除缓存文件。",
            approval_request=approval,
        )

        standard = result.to_standard_agent_result()

        self.assertTrue(standard.requires_user_action)
        self.assertEqual(["Approve or reject filesystem.delete_path"], standard.next_actions)
        self.assertEqual("filesystem.delete_path", standard.raw_result["approval_request"]["tool_name"])


if __name__ == "__main__":
    unittest.main()
