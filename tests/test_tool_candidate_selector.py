import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class ToolCandidateSelectorTest(unittest.TestCase):
    def test_selects_web_search_for_realtime_public_question(self) -> None:
        from app.agent_runtime.tool_candidate_selector import ToolCandidateSelector
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, create_default_agent_tool_registry

        selection = ToolCandidateSelector(create_default_agent_tool_registry()).select(
            "给我查一下梅西今天的比赛"
        )

        self.assertIn(EXTERNAL_WEB_SEARCH_TOOL, selection.capabilities)
        self.assertIn("realtime_public_information", selection.signals)

    def test_selects_web_search_for_colloquial_this_week_match_question(self) -> None:
        from app.agent_runtime.tool_candidate_selector import ToolCandidateSelector
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, create_default_agent_tool_registry

        selection = ToolCandidateSelector(create_default_agent_tool_registry()).select(
            "你看一下c罗这个星期有什么比赛吗"
        )

        self.assertIn(EXTERNAL_WEB_SEARCH_TOOL, selection.capabilities)
        self.assertIn("realtime_public_information", selection.signals)

    def test_selects_local_company_database_for_local_company_question(self) -> None:
        from app.agent_runtime.tool_candidate_selector import ToolCandidateSelector
        from app.agent_runtime.tool_registry import LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL, create_default_agent_tool_registry

        selection = ToolCandidateSelector(create_default_agent_tool_registry()).select(
            "数据库里有哪些公司，给我20个"
        )

        self.assertEqual((LOCAL_COMPANY_DATABASE_OVERVIEW_TOOL,), selection.capabilities)
        self.assertIn("local_company_data", selection.signals)

    def test_selects_local_job_source_overview_for_source_board_question(self) -> None:
        from app.agent_runtime.tool_candidate_selector import ToolCandidateSelector
        from app.agent_runtime.tool_registry import LOCAL_JOB_SOURCE_OVERVIEW_TOOL, create_default_agent_tool_registry

        selection = ToolCandidateSelector(create_default_agent_tool_registry()).select(
            "岗位来源库现在有多少条，给我20个"
        )

        self.assertEqual((LOCAL_JOB_SOURCE_OVERVIEW_TOOL,), selection.capabilities)
        self.assertIn("local_job_source_data", selection.signals)

    def test_selects_declared_agent_capability_from_capability_registry(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentCapabilityDefinition, AgentCapabilityRegistry
        from app.agent_runtime.tool_candidate_selector import ToolCandidateSelector
        from app.agent_runtime.tool_registry import AgentToolCandidateProfile

        registry = AgentCapabilityRegistry(
            [
                AgentCapabilityDefinition(
                    capability_id="resume.tailor",
                    name="简历优化",
                    description="根据目标岗位优化简历表达。",
                    executor_id="openai-sdk-agent",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    risk_level="low",
                    allowed_source_types=frozenset({"agent_chat"}),
                    candidate_profile=AgentToolCandidateProfile(
                        categories=frozenset({"resume_tailoring", "content_processing"}),
                        keywords=frozenset({"优化简历", "改简历", "匹配 JD"}),
                        examples=("帮我优化这段简历，让它更适合腾讯后端岗位",),
                    ),
                )
            ]
        )

        selection = ToolCandidateSelector(registry).select("帮我优化这段简历，让它更适合腾讯后端岗位")

        self.assertEqual(("resume.tailor",), selection.capabilities)
        self.assertIn("resume_tailoring", selection.signals)

    def test_does_not_force_tool_for_plain_writing_request(self) -> None:
        from app.agent_runtime.tool_candidate_selector import ToolCandidateSelector
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry

        selection = ToolCandidateSelector(create_default_agent_tool_registry()).select("帮我写一段自我介绍")

        self.assertEqual((), selection.capabilities)

    def test_selects_xiaohongshu_search_from_declared_content_source_tool(self) -> None:
        from app.agent_runtime.tool_candidate_selector import ToolCandidateSelector
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry

        selection = ToolCandidateSelector(create_default_agent_tool_registry()).select("请在小红书搜索 2027 秋招 Java 岗位")

        self.assertIn("xiaohongshu-mcp.search_feeds", selection.capabilities)
        self.assertIn("xiaohongshu_content_search", selection.signals)


if __name__ == "__main__":
    unittest.main()
