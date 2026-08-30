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

    def test_selects_database_company_profile_for_specific_company_question(self) -> None:
        from app.agent_runtime.tool_candidate_selector import ToolCandidateSelector
        from app.agent_runtime.tool_registry import DATABASE_COMPANY_PROFILE_TOOL, create_default_agent_tool_registry

        selection = ToolCandidateSelector(create_default_agent_tool_registry()).select(
            "数据库中关于京东这个公司的详细信息有什么"
        )

        self.assertEqual((DATABASE_COMPANY_PROFILE_TOOL,), selection.capabilities)
        self.assertIn("local_company_profile", selection.signals)

    def test_selects_database_job_search_for_local_job_question(self) -> None:
        from app.agent_runtime.tool_candidate_selector import ToolCandidateSelector
        from app.agent_runtime.tool_registry import DATABASE_JOB_SEARCH_TOOL, create_default_agent_tool_registry

        selection = ToolCandidateSelector(create_default_agent_tool_registry()).select(
            "查一下数据库里腾讯的 Python 后端岗位"
        )

        self.assertEqual((DATABASE_JOB_SEARCH_TOOL,), selection.capabilities)
        self.assertIn("local_job_search", selection.signals)

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

    def test_selects_filesystem_read_tool_for_user_provided_file_path(self) -> None:
        from app.agent_runtime.tool_candidate_selector import ToolCandidateSelector
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry

        selection = ToolCandidateSelector(create_default_agent_tool_registry()).select(
            "请读取 C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex",
            auto_executable_only=False,
        )

        self.assertIn("filesystem.read_file", selection.capabilities)
        self.assertIn("filesystem_read", selection.signals)

    def test_selects_filesystem_read_tool_for_can_you_read_path_question(self) -> None:
        from app.agent_runtime.tool_candidate_selector import ToolCandidateSelector
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry

        selection = ToolCandidateSelector(create_default_agent_tool_registry()).select(
            "你现在能不能读到 C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex 这个文件呢？",
            auto_executable_only=False,
        )

        self.assertIn("filesystem.read_file", selection.capabilities)
        self.assertIn("filesystem_read", selection.signals)

    def test_selects_filesystem_read_and_write_for_exact_file_text_replacement(self) -> None:
        from app.agent_runtime.tool_candidate_selector import ToolCandidateSelector
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry

        selection = ToolCandidateSelector(create_default_agent_tool_registry()).select(
            "请把 C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex 里的刘汉卿替换为王爷，其他不要动",
            auto_executable_only=False,
        )

        self.assertIn("filesystem.read_file", selection.capabilities)
        self.assertIn("filesystem.write_text", selection.capabilities)
        self.assertIn("filesystem.replace_text", selection.capabilities)
        self.assertIn("filesystem_read", selection.signals)
        self.assertIn("filesystem_write", selection.signals)
        self.assertIn("filesystem_replace", selection.signals)
        self.assertNotIn("filesystem.delete_path", selection.capabilities)
        self.assertNotIn("filesystem.move_file", selection.capabilities)


if __name__ == "__main__":
    unittest.main()
