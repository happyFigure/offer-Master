import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class FakePlannerLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = []

    def complete(self, *, messages):
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        self.messages.append(messages)
        return LLMChatCompletion(content=self.content)


class ExecutionPlannerTest(unittest.TestCase):
    def _campus_search_context_pack(self) -> dict[str, object]:
        from app.agent_runtime.context.capability_catalog import CapabilityCatalog
        from app.agent_runtime.context.context_pack import ContextPackBuilder
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry
        from app.agent_runtime.understanding.schemas import EntityFrame, IntentFrame

        frame = IntentFrame(
            intent="campus_recruiting_search",
            confidence=0.96,
            needs_external_info=True,
            risk_level="low",
            entities=EntityFrame(company_names=["经纬恒润"], keywords=["校园招聘"], time_range="latest"),
        )
        return ContextPackBuilder(CapabilityCatalog.from_registry(create_default_agent_tool_registry())).build(frame).to_metadata_dict()

    def test_llm_execution_planner_returns_capability_action(self) -> None:
        from app.agent_runtime.planning.execution_planner import HybridExecutionPlanner
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL

        llm = FakePlannerLLM(
            """
            ```json
            {
              "mode": "simple_tool_call",
              "confidence": 0.91,
              "risk_level": "low",
              "actions": [
                {
                  "type": "call_capability",
                  "capability": "external.web_search",
                  "arguments": {
                    "query": "经纬恒润 校园招聘 官网",
                    "max_results": 5
                  },
                  "reason": "需要查询最新校招入口"
                }
              ],
              "reason": "用户要求查询最新校招信息"
            }
            ```
            """
        )

        plan = HybridExecutionPlanner(llm_client=llm).plan(
            user_message="查一下经纬恒润的校园招聘信息",
            context_pack=self._campus_search_context_pack(),
        )

        action = plan.primary_action()
        self.assertEqual("simple_tool_call", plan.mode)
        self.assertIsNotNone(action)
        self.assertEqual("call_capability", action.type)
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, action.capability)
        self.assertEqual({"query": "经纬恒润 校园招聘 官网", "max_results": 5}, action.arguments)
        self.assertIn("只能输出 JSON", llm.messages[0][0]["content"])

    def test_execution_planner_blocks_capability_outside_context_pack(self) -> None:
        from app.agent_runtime.planning.execution_planner import HybridExecutionPlanner

        llm = FakePlannerLLM(
            """
            {
              "mode": "simple_tool_call",
              "confidence": 0.88,
              "risk_level": "medium",
              "actions": [
                {
                  "type": "call_capability",
                  "capability": "offerio.sync_company_jobs",
                  "arguments": {"limit": 1000}
                }
              ]
            }
            """
        )

        plan = HybridExecutionPlanner(llm_client=llm).plan(
            user_message="查一下经纬恒润的校园招聘信息",
            context_pack=self._campus_search_context_pack(),
        )

        action = plan.primary_action()
        self.assertEqual("blocked", plan.mode)
        self.assertIsNotNone(action)
        self.assertEqual("final_answer", action.type)
        self.assertIn("outside ContextPack", plan.reason or "")


if __name__ == "__main__":
    unittest.main()
