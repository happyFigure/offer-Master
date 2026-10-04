import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class ToolSemanticProfileTest(unittest.TestCase):
    def test_database_company_list_tool_declares_runtime_semantic_profile(self) -> None:
        from app.agent_runtime.tool_registry import DATABASE_COMPANY_LIST_TOOL, create_default_agent_tool_registry

        definition = create_default_agent_tool_registry().get(DATABASE_COMPANY_LIST_TOOL)

        self.assertIsNotNone(definition)
        profile = definition.semantic_profile
        self.assertIsNotNone(profile)
        self.assertEqual("list_local_companies", profile.intent)
        self.assertEqual("company_list", profile.target_type)
        self.assertEqual((), profile.modifies)
        self.assertIn("local_database", profile.does_not_modify)
        self.assertIn("tool_result_ok", profile.success_criteria)
        self.assertIn("result_field_present:companies", profile.success_criteria)
        self.assertIn("query local database with narrower keyword", profile.failure_recovery)
        self.assertEqual(["result.companies"], profile.result_contract["required_fields"])

    def test_filesystem_rename_tool_declares_positive_and_negative_boundaries(self) -> None:
        from app.agent_runtime.tool_registry import FILESYSTEM_MOVE_FILE_TOOL, create_default_agent_tool_registry

        definition = create_default_agent_tool_registry().get(FILESYSTEM_MOVE_FILE_TOOL)

        self.assertIsNotNone(definition)
        profile = definition.candidate_profile
        self.assertIsNotNone(profile)
        self.assertIn("用户要改变文件名、文件路径或移动位置", profile.use_when)
        self.assertIn("用户要修改文件正文里的姓名、标题或项目内容", profile.do_not_use_when)
        self.assertIn("把这个文件名改成刘汉卿-后端开发-AI-Agent.tex", profile.positive_examples)
        self.assertIn("把简历里的刘汉卿改成王爷", profile.negative_examples)
        self.assertIn("filename", profile.required_context_focus)
        self.assertTrue(any("文件名" in note for note in profile.disambiguation_notes))

    def test_runtime_panel_exposes_tool_semantic_boundaries(self) -> None:
        from app.api.v1.agent_runtime import _serialize_capability
        from app.agent_runtime.agent_as_tool import create_default_agent_capability_registry
        from app.agent_runtime.tool_registry import FILESYSTEM_MOVE_FILE_TOOL, create_default_agent_tool_registry
        from app.core.config import Settings

        tool_registry = create_default_agent_tool_registry()
        capability_registry = create_default_agent_capability_registry(tool_registry=tool_registry)
        capability = capability_registry.get(FILESYSTEM_MOVE_FILE_TOOL)

        self.assertIsNotNone(capability)
        payload = _serialize_capability(capability, settings=Settings())

        self.assertIn("用户要改变文件名、文件路径或移动位置", payload["candidate_use_when"])
        self.assertIn("用户要修改文件正文里的姓名、标题或项目内容", payload["candidate_do_not_use_when"])
        self.assertIn("把这个文件名改成刘汉卿-后端开发-AI-Agent.tex", payload["candidate_positive_examples"])
        self.assertIn("把简历里的刘汉卿改成王爷", payload["candidate_negative_examples"])
        self.assertEqual("rename_or_move_file", payload["semantic_profile"]["intent"])
        self.assertEqual("file_path", payload["semantic_profile"]["target_type"])
        self.assertIn("dst_path_exists", payload["semantic_profile"]["success_criteria"])

    def test_generic_goal_validator_uses_tool_semantic_profile_result_contract(self) -> None:
        from app.agent_runtime.goal.schemas import GoalState
        from app.agent_runtime.goal.validator import validate_goal_completion
        from app.agent_runtime.tool_registry import AgentToolSemanticProfile, DATABASE_COMPANY_LIST_TOOL

        semantic_profile = AgentToolSemanticProfile(
            intent="list_local_companies",
            target_type="company_list",
            success_criteria=("tool_result_ok", "result_field_present:companies"),
            failure_recovery=("query local database with narrower keyword",),
            result_contract={"required_fields": ["result.companies"]},
        )
        goal = GoalState(
            intent="local_company_database_list",
            target_type="company_list",
            expected_operation="list_local_companies",
            success_criteria=("tool_result_ok", "result_field_present:companies"),
        )

        completed = validate_goal_completion(
            goal_state=goal,
            tool_name=DATABASE_COMPANY_LIST_TOOL,
            tool_input={"limit": 20},
            result_payload={"ok": True, "result": {"companies": [{"company_name": "Tencent"}]}},
            semantic_profile=semantic_profile,
        )
        missing_contract = validate_goal_completion(
            goal_state=goal,
            tool_name=DATABASE_COMPANY_LIST_TOOL,
            tool_input={"limit": 20},
            result_payload={"ok": True, "result": {"summary": "ok"}},
            semantic_profile=semantic_profile,
        )
        wrong_tool = validate_goal_completion(
            goal_state=goal,
            tool_name="database.company_search",
            tool_input={"company_names": ["Tencent"]},
            result_payload={"ok": True, "result": {"companies": []}},
            semantic_profile=AgentToolSemanticProfile(intent="search_local_companies", target_type="company_profile"),
        )

        self.assertTrue(completed.completed)
        self.assertTrue(completed.advanced)
        self.assertEqual("final_answer", completed.next_action)
        self.assertFalse(missing_contract.completed)
        self.assertTrue(missing_contract.advanced)
        self.assertEqual("continue_loop", missing_contract.next_action)
        self.assertIn("result.companies", missing_contract.missing_information)
        self.assertFalse(wrong_tool.completed)
        self.assertFalse(wrong_tool.advanced)
        self.assertEqual("continue_loop", wrong_tool.next_action)
        self.assertEqual("list_local_companies", wrong_tool.suggested_operation)


if __name__ == "__main__":
    unittest.main()
