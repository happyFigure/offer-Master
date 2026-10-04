import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class FinalAnswerContractTest(unittest.TestCase):
    def test_summary_request_builds_non_raw_answer_contract(self) -> None:
        from app.agent_runtime.final_answer.contracts import build_answer_contract
        from app.agent_runtime.goal.schemas import GoalState

        goal_state = GoalState(
            intent="filesystem_operation",
            target_type="file",
            expected_operation="read_file",
            target={"path": "C:/resume.tex"},
            answer_intent="summarize_document",
            answer_policy={"language": "zh-CN"},
        )

        contract = build_answer_contract(
            user_message="你给我总结一下这个简历内容",
            goal_state=goal_state,
            context_metadata={"filesystem_operation": "read_file"},
        )

        self.assertEqual("summarize_document", contract.answer_intent)
        self.assertEqual(["read_file"], contract.evidence_actions)
        self.assertFalse(contract.allow_raw_tool_output)
        self.assertTrue(contract.answer_policy["do_not_echo_full_content"])
        self.assertEqual("final_answer_quality", contract.completion_check["kind"])

    def test_explicit_show_content_allows_raw_tool_output(self) -> None:
        from app.agent_runtime.final_answer.contracts import build_answer_contract

        contract = build_answer_contract(
            user_message="显示这个文件的全文",
            goal_state=None,
            context_metadata={"filesystem_operation": "read_file"},
        )

        self.assertEqual("show_content", contract.answer_intent)
        self.assertTrue(contract.allow_raw_tool_output)
        self.assertFalse(contract.answer_policy.get("do_not_echo_full_content", False))

    def test_contract_round_trips_through_metadata(self) -> None:
        from app.agent_runtime.final_answer.contracts import AnswerContract

        original = AnswerContract(
            answer_intent="analyze_document",
            evidence_actions=["read_file"],
            answer_policy={"language": "zh-CN", "do_not_echo_full_content": True},
            completion_check={"kind": "final_answer_quality"},
            allow_raw_tool_output=False,
        )

        restored = AnswerContract.from_metadata_dict(original.to_metadata_dict())

        self.assertEqual(original, restored)


if __name__ == "__main__":
    unittest.main()
