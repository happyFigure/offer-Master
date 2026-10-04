import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class GoalValidationLoopRecoveryTest(unittest.TestCase):
    def test_goal_validation_marks_successful_tool_result_as_partial_when_goal_is_unfinished(self) -> None:
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY
        from app.agent_runtime.graph_factory import _loop_agent_observation_status, _loop_agent_observation_summary
        from app.agent_runtime.state import AgentState

        payload = {
            "tool_name": FILESYSTEM_SKILL_CAPABILITY,
            "status": "succeeded",
            "result": {
                "ok": True,
                "operation": "read_file",
                "goal_validation": {
                    "completed": False,
                    "advanced": False,
                    "recoverable": True,
                    "reason": "用户目标是重命名文件，但实际执行的是 read_file。",
                    "next_action": "continue_loop",
                    "suggested_capability": FILESYSTEM_SKILL_CAPABILITY,
                    "suggested_operation": "rename_file",
                    "missing_information": [],
                },
            },
        }
        state = AgentState(
            session_id="s",
            workflow_run_id="w",
            agent_run_id="a",
            user_message="把文件名改成 B.tex",
            current_step="maybe_tool",
        )

        self.assertEqual("partial", _loop_agent_observation_status(payload, state))
        self.assertIn("重命名文件", _loop_agent_observation_summary(FILESYSTEM_SKILL_CAPABILITY, payload, state))

    def test_goal_validation_builds_suggested_recovery_decision_for_continue_loop(self) -> None:
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY
        from app.agent_runtime.graph_factory import _goal_recovery_decision_from_validation
        from app.agent_runtime.loop_agent.schemas import LoopAgentAction

        payload = {
            "result": {
                "goal_validation": {
                    "completed": False,
                    "advanced": False,
                    "recoverable": True,
                    "reason": "用户目标是重命名文件，但实际执行的是 read_file。",
                    "next_action": "continue_loop",
                    "suggested_capability": FILESYSTEM_SKILL_CAPABILITY,
                    "suggested_operation": "rename_file",
                    "missing_information": [],
                }
            }
        }

        decision = _goal_recovery_decision_from_validation(
            FILESYSTEM_SKILL_CAPABILITY,
            {"user_task": "把文件名字改成 B.tex"},
            payload,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(LoopAgentAction.CALL_TOOL, decision.action)
        self.assertEqual(FILESYSTEM_SKILL_CAPABILITY, decision.capability)
        self.assertEqual("rename_file", decision.tool_input["operation"])
        self.assertTrue(decision.tool_input["goal_recovery"])
        self.assertEqual("goal_validation", decision.metadata["source"])

    def test_no_dead_end_retry_outcome_builds_suggested_recovery_decision(self) -> None:
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY
        from app.agent_runtime.graph_factory import _no_dead_end_recovery_decision_from_payload
        from app.agent_runtime.loop_agent.schemas import LoopAgentAction

        payload = {
            "result": {
                "ok": False,
                "operation": "read_file",
                "recoverable": True,
                "next_action": "retry_operation",
                "suggested_operation": "path_exists",
                "reason": "用户目标是检查文件是否存在，但 read_file 没有推进目标。",
            }
        }

        decision = _no_dead_end_recovery_decision_from_payload(
            FILESYSTEM_SKILL_CAPABILITY,
            {"user_task": "看下这个文件是否村"},
            payload,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(LoopAgentAction.CALL_TOOL, decision.action)
        self.assertEqual(FILESYSTEM_SKILL_CAPABILITY, decision.capability)
        self.assertEqual("path_exists", decision.tool_input["operation"])
        self.assertTrue(decision.tool_input["no_dead_end_recovery"])
        self.assertEqual("no_dead_end", decision.metadata["source"])


if __name__ == "__main__":
    unittest.main()
