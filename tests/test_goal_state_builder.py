import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class GoalStateBuilderTest(unittest.TestCase):
    def test_local_company_list_intent_builds_semantic_tool_goal(self) -> None:
        from app.agent_runtime.goal.builder import build_goal_state

        goal = build_goal_state(
            user_message="数据库里有哪些公司？给我列出来",
            intent="local_company_database_list",
            context_metadata={},
        )

        self.assertEqual("local_company_database_list", goal.intent)
        self.assertEqual("company_list", goal.target_type)
        self.assertEqual("list_local_companies", goal.expected_operation)
        self.assertIn("tool_result_ok", goal.success_criteria)

    def test_rename_goal_requires_structured_name_intent(self) -> None:
        from app.agent_runtime.goal.builder import build_goal_state

        source_path = r"C:\简历\一下.tex"
        goal = build_goal_state(
            user_message="把名字修改为你起的名字",
            intent="filesystem_operation",
            context_metadata={
                "filesystem_operation": "rename_file",
                "active_file": {"path": source_path},
                "operation_intent": {
                    "destination": {"kind": "directory", "path": r"C:\简历"},
                    "name_intent": {
                        "display_name": "LiuHanqing_AI-Agent-Java-Engineer_Resume_CN.tex",
                        "source_basis": "file_content",
                    },
                },
            },
        )

        self.assertEqual("rename_file", goal.expected_operation)
        self.assertEqual("LiuHanqing_AI-Agent-Java-Engineer_Resume_CN.tex", goal.target["target_name"])
        self.assertIn("operation_is_rename", goal.success_criteria)

    def test_copy_goal_requires_structured_destination(self) -> None:
        from app.agent_runtime.goal.builder import build_goal_state

        source_path = r"C:\简历\一下.tex"
        target_path = r"C:\简历\一下-副本.tex"
        goal = build_goal_state(
            user_message="给我复制一下",
            intent="filesystem_operation",
            context_metadata={
                "filesystem_operation": "copy_file",
                "active_file": {"path": source_path},
                "operation_intent": {
                    "destination": {"kind": "file", "path": target_path},
                },
            },
        )

        self.assertEqual("copy_file", goal.expected_operation)
        self.assertEqual(target_path, goal.target["target_path"])

    def test_path_exists_goal_uses_structured_operation_and_active_file(self) -> None:
        from app.agent_runtime.goal.builder import build_goal_state

        path = r"C:\简历\刘汉卿.tex"
        goal = build_goal_state(
            user_message="这个文件是否村",
            intent="filesystem_operation",
            context_metadata={
                "filesystem_operation": "path_exists",
                "active_file": {"path": path},
            },
        )

        self.assertEqual("path_exists", goal.expected_operation)
        self.assertEqual(path, goal.target["path"])

    def test_read_goal_keeps_summary_as_answer_policy(self) -> None:
        from app.agent_runtime.goal.builder import build_goal_state

        path = r"C:\简历\刘汉卿.tex"
        goal = build_goal_state(
            user_message="给我总结一下这个文件的内容",
            intent="filesystem_operation",
            context_metadata={
                "filesystem_operation": "read_file",
                "active_file": {"path": path},
            },
        )

        self.assertEqual("read_file", goal.expected_operation)
        self.assertEqual("summarize_document", goal.answer_intent)
        self.assertTrue(goal.answer_policy["do_not_echo_full_content"])


if __name__ == "__main__":
    unittest.main()
