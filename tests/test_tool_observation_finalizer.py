import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class ToolObservationFinalizerTest(unittest.TestCase):
    def test_extracts_tool_observation_from_runtime_transcript(self) -> None:
        from app.agent_runtime.final_answer.observations import extract_tool_observations
        from app.agent_runtime.state import AgentState

        state = AgentState(
            session_id="session-1",
            workflow_run_id="workflow-1",
            agent_run_id="agent-run-1",
            user_message="分析这个简历和 Java 后端岗位的匹配度",
            current_step="final_response",
            requested_tool_name="skill.resume_match",
            tool_call_ids=["tool-call-1"],
            llm_messages=[
                {
                    "role": "assistant",
                    "content": "Tool result: skill.resume_match succeeded",
                    "metadata": {
                        "source": "tool_transcript",
                        "content_json": {
                            "tool_name": "skill.resume_match",
                            "status": "succeeded",
                            "result": {
                                "operation": "match_jd",
                                "summary": "匹配度 82 分",
                                "result": {"score": 82, "matched_skills": ["Java", "Spring Boot"]},
                            },
                            "error": None,
                        },
                    },
                }
            ],
        )

        observations = extract_tool_observations(state)

        self.assertEqual(1, len(observations))
        self.assertEqual("skill.resume_match", observations[0].tool_name)
        self.assertEqual("match_jd", observations[0].operation)
        self.assertTrue(observations[0].ok)
        self.assertIn("Spring Boot", observations[0].evidence)

    def test_builds_generic_final_answer_synthesis_request(self) -> None:
        from app.agent_runtime.final_answer.synthesis import build_tool_observation_final_answer_request
        from app.agent_runtime.state import AgentState

        state = AgentState(
            session_id="session-1",
            workflow_run_id="workflow-1",
            agent_run_id="agent-run-1",
            user_message="总结这个 PDF 解析结果",
            current_step="final_response",
            requested_tool_name="skill.pdf_parser",
            tool_call_ids=["tool-call-1"],
            llm_messages=[
                {
                    "role": "assistant",
                    "content": "Tool result: skill.pdf_parser succeeded",
                    "metadata": {
                        "source": "tool_transcript",
                        "content_json": {
                            "tool_name": "skill.pdf_parser",
                            "status": "succeeded",
                            "result": {"operation": "extract_text", "result": {"content": "第一页：项目经历和技术栈"}},
                        },
                    },
                }
            ],
        )

        request = build_tool_observation_final_answer_request(state)

        self.assertIsNotNone(request)
        assert request is not None
        combined = "\n".join(str(message.get("content") or "") for message in request.messages)
        self.assertEqual("llm_tool_observation_final_answer", request.response_mode)
        self.assertEqual("summarize_document", request.contract.answer_intent)
        self.assertIn("工具结果只是 observation", combined)
        self.assertIn("AnswerContract", combined)
        self.assertIn("总结这个 PDF 解析结果", combined)
        self.assertIn("第一页：项目经历和技术栈", combined)

    def test_validator_rejects_status_only_or_raw_tool_output_for_summary(self) -> None:
        from app.agent_runtime.final_answer.contracts import AnswerContract
        from app.agent_runtime.final_answer.validator import validate_final_answer

        contract = AnswerContract(
            answer_intent="summarize_document",
            evidence_actions=["read_file"],
            answer_policy={"do_not_echo_full_content": True},
            completion_check={"kind": "final_answer_quality"},
            allow_raw_tool_output=False,
        )

        self.assertFalse(validate_final_answer("已读取文件 C:/resume.tex。", contract=contract, observations=[]).passed)
        self.assertFalse(validate_final_answer("```text\n\\documentclass{article}\n```", contract=contract, observations=[]).passed)
        self.assertTrue(validate_final_answer("这份简历主要包含教育背景、项目经历和技术栈。", contract=contract, observations=[]).passed)


if __name__ == "__main__":
    unittest.main()
