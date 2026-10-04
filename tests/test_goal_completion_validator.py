from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class GoalCompletionValidatorTest(unittest.TestCase):
    def test_read_file_success_does_not_complete_rename_goal(self) -> None:
        from app.agent_runtime.goal.schemas import GoalState
        from app.agent_runtime.goal.validator import validate_goal_completion

        goal = GoalState(
            intent="filesystem_operation",
            target_type="file_name",
            expected_operation="rename_file",
            target={"source_path": "A.tex", "target_name": "B.tex"},
            success_criteria=("operation_is_rename", "target_name_matches", "tool_result_ok"),
        )

        result = validate_goal_completion(
            goal_state=goal,
            tool_name="skill.filesystem",
            tool_input={"user_task": "把文件名改成 B.tex"},
            result_payload={"tool_name": "skill.filesystem", "ok": True, "operation": "read_file"},
        )

        self.assertFalse(result.completed)
        self.assertFalse(result.advanced)
        self.assertTrue(result.recoverable)
        self.assertEqual("continue_loop", result.next_action)
        self.assertEqual("skill.filesystem", result.suggested_capability)
        self.assertEqual("rename_file", result.suggested_operation)
        self.assertIn("重命名", result.reason)

    def test_replace_text_zero_replacements_does_not_complete_content_goal(self) -> None:
        from app.agent_runtime.goal.schemas import GoalState
        from app.agent_runtime.goal.validator import validate_goal_completion

        goal = GoalState(
            intent="filesystem_operation",
            target_type="file_content",
            expected_operation="replace_text",
            target={"path": "resume.tex", "old_text": "Spring Cloud", "new_text": "Spring AI"},
            success_criteria=("operation_is_replace_text", "replacement_count_positive", "tool_result_ok"),
        )

        result = validate_goal_completion(
            goal_state=goal,
            tool_name="skill.filesystem",
            tool_input={"user_task": "把 Spring Cloud 改成 Spring AI"},
            result_payload={"tool_name": "skill.filesystem", "ok": True, "operation": "replace_text", "replacement_count": 0},
        )

        self.assertFalse(result.completed)
        self.assertTrue(result.recoverable)
        self.assertEqual("ask_user_or_search_text", result.next_action)
        self.assertEqual("replace_text", result.suggested_operation)
        self.assertIn("替换次数为 0", result.reason)

    def test_replace_text_noop_completes_when_target_already_satisfied(self) -> None:
        from app.agent_runtime.goal.schemas import GoalState
        from app.agent_runtime.goal.validator import validate_goal_completion

        goal = GoalState(
            intent="filesystem_operation",
            target_type="file_content",
            expected_operation="replace_text",
            target={"path": "resume.tex", "old_text": "刘汉卿", "new_text": "刘汉卿"},
            success_criteria=("operation_is_replace_text", "replacement_count_positive", "tool_result_ok"),
        )

        result = validate_goal_completion(
            goal_state=goal,
            tool_name="skill.filesystem",
            tool_input={"user_task": "把名字改成刘汉卿"},
            result_payload={"tool_name": "skill.filesystem", "ok": True, "operation": "replace_text", "replacement_count": 0, "no_op": True},
        )

        self.assertTrue(result.completed)
        self.assertTrue(result.advanced)
        self.assertFalse(result.recoverable)
        self.assertEqual("final_answer", result.next_action)
        self.assertIn("无需再次执行写操作", result.reason)

    def test_confirmed_rename_matching_target_completes_goal(self) -> None:
        from app.agent_runtime.goal.schemas import GoalState
        from app.agent_runtime.goal.validator import validate_goal_completion

        goal = GoalState(
            intent="filesystem_operation",
            target_type="file_name",
            expected_operation="rename_file",
            target={"source_path": "C:/tmp/A.tex", "target_name": "B.tex"},
            success_criteria=("operation_is_rename", "target_name_matches", "tool_result_ok"),
        )

        result = validate_goal_completion(
            goal_state=goal,
            tool_name="skill.filesystem",
            tool_input={"user_task": "把文件名改成 B.tex"},
            result_payload={"tool_name": "skill.filesystem", "ok": True, "operation": "rename_file", "dst": "C:/tmp/B.tex"},
        )

        self.assertTrue(result.completed)
        self.assertTrue(result.advanced)
        self.assertFalse(result.recoverable)
        self.assertEqual("final_answer", result.next_action)

    def test_rename_postcheck_failure_does_not_complete_goal_even_when_ok_true(self) -> None:
        from app.agent_runtime.goal.schemas import GoalState
        from app.agent_runtime.goal.validator import validate_goal_completion

        goal = GoalState(
            intent="filesystem_operation",
            target_type="file_name",
            expected_operation="rename_file",
            target={"source_path": "C:/tmp/A.tex", "target_name": "B.tex"},
            success_criteria=("operation_is_rename", "target_name_matches", "tool_result_ok"),
        )

        result = validate_goal_completion(
            goal_state=goal,
            tool_name="skill.filesystem",
            tool_input={"user_task": "把文件名改成 B.tex"},
            result_payload={
                "tool_name": "skill.filesystem",
                "ok": True,
                "operation": "rename_file",
                "arguments": {"src": "C:/tmp/A.tex", "dst": "C:/tmp/B.tex"},
                "filesystem_trace": {
                    "postcheck": {
                        "completed": False,
                        "reason": "target_missing_after_rename",
                        "source_exists_after": True,
                        "target_exists_after": False,
                    }
                },
            },
        )

        self.assertFalse(result.completed)
        self.assertFalse(result.advanced)
        self.assertTrue(result.recoverable)
        self.assertEqual("continue_loop", result.next_action)
        self.assertIn("复核未通过", result.reason)

    def test_path_exists_result_completes_existence_question_even_when_missing(self) -> None:
        from app.agent_runtime.goal.schemas import GoalState
        from app.agent_runtime.goal.validator import validate_goal_completion

        goal = GoalState(
            intent="filesystem_operation",
            target_type="file_path",
            expected_operation="path_exists",
            target={"path": "C:/tmp/missing.tex"},
            success_criteria=("operation_is_path_exists", "tool_result_ok"),
        )

        result = validate_goal_completion(
            goal_state=goal,
            tool_name="skill.filesystem",
            tool_input={"user_task": "看下这个文件是否存在"},
            result_payload={"tool_name": "skill.filesystem", "ok": True, "operation": "path_exists", "result": {"result": {"exists": False}}},
        )

        self.assertTrue(result.completed)
        self.assertTrue(result.advanced)
        self.assertFalse(result.recoverable)
        self.assertEqual("final_answer", result.next_action)

    def test_copy_file_matching_target_completes_goal(self) -> None:
        from app.agent_runtime.goal.schemas import GoalState
        from app.agent_runtime.goal.validator import validate_goal_completion

        goal = GoalState(
            intent="filesystem_operation",
            target_type="file_copy",
            expected_operation="copy_file",
            target={"source_path": "C:/tmp/A.tex", "target_path": "C:/tmp/B.tex"},
            success_criteria=("operation_is_copy_file", "tool_result_ok", "target_path_matches"),
        )

        result = validate_goal_completion(
            goal_state=goal,
            tool_name="skill.filesystem",
            tool_input={"user_task": "复制一份叫 B.tex"},
            result_payload={
                "tool_name": "skill.filesystem",
                "ok": True,
                "operation": "copy_file",
                "arguments": {"src": "C:/tmp/A.tex", "dst": "C:/tmp/B.tex"},
            },
        )

        self.assertTrue(result.completed)
        self.assertTrue(result.advanced)
        self.assertFalse(result.recoverable)
        self.assertEqual("final_answer", result.next_action)

    def test_unknown_operation_does_not_complete_copy_goal(self) -> None:
        from app.agent_runtime.goal.schemas import GoalState
        from app.agent_runtime.goal.validator import validate_goal_completion

        goal = GoalState(
            intent="filesystem_operation",
            target_type="file_copy",
            expected_operation="copy_file",
            target={"source_path": "C:/tmp/A.tex", "target_path": "C:/tmp/B.tex"},
            success_criteria=("operation_is_copy_file", "tool_result_ok", "target_path_matches"),
        )

        result = validate_goal_completion(
            goal_state=goal,
            tool_name="skill.filesystem",
            tool_input={"user_task": "复制一份叫 B.tex"},
            result_payload={"tool_name": "skill.filesystem", "ok": False, "operation": "unknown"},
        )

        self.assertFalse(result.completed)
        self.assertFalse(result.advanced)
        self.assertTrue(result.recoverable)
        self.assertEqual("continue_loop", result.next_action)
        self.assertEqual("copy_file", result.suggested_operation)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
