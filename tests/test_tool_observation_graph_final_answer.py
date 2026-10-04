import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class ToolObservationGraphFinalAnswerTest(unittest.TestCase):
    def _state_with_resume_match_observation(self):
        from app.agent_runtime.state import AgentState

        return AgentState(
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
                                "result": {"score": 82, "gaps": ["项目量化不足"], "matched_skills": ["Java", "Spring Boot"]},
                            },
                            "error": None,
                        },
                    },
                }
            ],
        )

    def test_generate_final_response_uses_generic_synthesis_for_unknown_skill_observation(self) -> None:
        from app.agent_runtime.graph_factory import _generate_final_response
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        expected = "匹配度约 82 分，优势是 Java 和 Spring Boot，主要短板是项目量化不足。"

        class FakeLLM:
            def __init__(self, test_case: ToolObservationGraphFinalAnswerTest) -> None:
                self.calls = []
                self._test_case = test_case

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append(messages)
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                self._test_case.assertIn("工具结果只是 observation", combined)
                self._test_case.assertIn("AnswerContract", combined)
                self._test_case.assertIn("分析这个简历和 Java 后端岗位的匹配度", combined)
                self._test_case.assertIn("Spring Boot", combined)
                return LLMChatCompletion(content=expected)

        llm = FakeLLM(self)
        response, mode = _generate_final_response(self._state_with_resume_match_observation(), dependencies=SimpleNamespace(llm_client=llm))

        self.assertEqual(expected, response)
        self.assertEqual("llm_tool_observation_final_answer", mode)
        self.assertEqual(1, len(llm.calls))

    def test_generate_final_response_retries_when_first_synthesis_is_tool_status_only(self) -> None:
        from app.agent_runtime.graph_factory import _generate_final_response
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeLLM:
            def __init__(self, test_case: ToolObservationGraphFinalAnswerTest) -> None:
                self.calls = []
                self._test_case = test_case

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append(messages)
                if len(self.calls) == 1:
                    return LLMChatCompletion(content="工具执行成功。")
                combined = "\n".join(str(message.get("content") or "") for message in messages)
                self._test_case.assertIn("上一次最终回答没有满足 AnswerContract", combined)
                return LLMChatCompletion(content="简历和岗位匹配度较高，Java/Spring Boot 是优势，建议补充项目量化指标。")

        llm = FakeLLM(self)
        response, mode = _generate_final_response(self._state_with_resume_match_observation(), dependencies=SimpleNamespace(llm_client=llm))

        self.assertIn("项目量化", response)
        self.assertEqual("llm_tool_observation_final_answer_retry", mode)
        self.assertEqual(2, len(llm.calls))


if __name__ == "__main__":
    unittest.main()
