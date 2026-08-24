import sys
import unittest
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class LoopAgentToolChoiceRunnerTest(unittest.TestCase):
    def test_runner_executes_model_selected_tool_and_finishes_from_observation(self) -> None:
        from app.agent_runtime.loop_agent.tool_choice_runner import LoopAgentTask, ToolChoiceLoopRunner
        from app.agent_runtime.loop_agent.schemas import LoopAgentStopReason
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        executed_inputs = []

        def fake_search(_session, *, query: str, max_results: int = 5):
            executed_inputs.append({"query": query, "max_results": max_results})
            return {
                "tool_name": "external.web_search",
                "ok": True,
                "result": {
                    "answer": "Canonical 是 Ubuntu 背后的公司，主要提供企业 Linux、云和安全支持。",
                    "sources": [{"title": "Canonical", "url": "https://canonical.com/"}],
                },
            }

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="external.web_search",
                    description="查询公开网页资料。",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "default": 5},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=fake_search,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            ]
        )

        class FakeLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                if len(self.calls) == 1:
                    test_case.assertEqual("auto", tool_choice)
                    test_case.assertEqual(["external_web_search"], [tool["function"]["name"] for tool in tools])
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-canonical-search",
                                name="external_web_search",
                                arguments={"query": "Canonical Ltd. 主要业务", "max_results": 3},
                            )
                        ],
                    )
                test_case.assertTrue(any("Canonical 是 Ubuntu 背后的公司" in message["content"] for message in messages))
                return LLMChatCompletion(content="Canonical Ltd. 主要做 Ubuntu、企业 Linux、云基础设施和安全支持。")

        test_case = self
        result = ToolChoiceLoopRunner(registry=registry, llm_client=FakeLLM()).run(
            LoopAgentTask(
                user_message="Canonical Ltd. 是做什么的？主要业务是什么？",
                available_capabilities=("external.web_search",),
                source_type="agent_chat",
            ),
            max_steps=2,
        )

        self.assertEqual(LoopAgentStopReason.MODEL_FINAL, result.stop_reason)
        self.assertEqual("Canonical Ltd. 主要做 Ubuntu、企业 Linux、云基础设施和安全支持。", result.final_answer)
        self.assertEqual([{"query": "Canonical Ltd. 主要业务", "max_results": 3}], executed_inputs)
        self.assertEqual(1, result.executed_step_count)
        self.assertEqual("external.web_search", result.trace[0].capability)
        self.assertEqual("succeeded", result.trace[0].observation_status)
        self.assertEqual("call-canonical-search", result.trace[0].tool_call_id)

    def test_runner_blocks_model_requested_tool_that_was_not_offered_this_turn(self) -> None:
        from app.agent_runtime.loop_agent.tool_choice_runner import LoopAgentTask, ToolChoiceLoopRunner
        from app.agent_runtime.loop_agent.schemas import LoopAgentStopReason
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        executed = []

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="external.web_search",
                    description="查询公开网页资料。",
                    input_schema={"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=lambda _session, **arguments: executed.append(arguments) or {"ok": True},
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            ]
        )

        class FakeLLM:
            def complete(self, *, messages, tools=None, tool_choice=None):
                return LLMChatCompletion(
                    content="",
                    tool_calls=[LLMToolCall(id="call-hidden-tool", name="external_web_search", arguments={"query": "腾讯校招"})],
                )

        result = ToolChoiceLoopRunner(registry=registry, llm_client=FakeLLM()).run(
            LoopAgentTask(
                user_message="查腾讯校招",
                available_capabilities=(),
                source_type="agent_chat",
            ),
            max_steps=2,
        )

        self.assertEqual(LoopAgentStopReason.STEP_FAILED, result.stop_reason)
        self.assertEqual([], executed)
        self.assertEqual("external_web_search", result.trace[0].capability)
        self.assertEqual("failed", result.trace[0].observation_status)
        self.assertEqual("TOOL_NOT_OFFERED", result.trace[0].metadata["observation"]["metadata"]["error_code"])

    def test_runner_converts_textual_tool_call_into_real_tool_execution(self) -> None:
        from app.agent_runtime.loop_agent.tool_choice_runner import LoopAgentTask, ToolChoiceLoopRunner
        from app.agent_runtime.loop_agent.schemas import LoopAgentStopReason
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        executed_inputs = []

        def fake_search(_session, *, query: str, max_results: int = 5):
            executed_inputs.append({"query": query, "max_results": max_results})
            return {
                "tool_name": "external.web_search",
                "ok": True,
                "result": {"answer": "C 罗本周赛程以官方公布为准。", "sources": []},
            }

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="external.web_search",
                    description="查询公开网页资料。",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=fake_search,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            ]
        )

        class FakeLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                if len(self.calls) == 1:
                    return LLMChatCompletion(
                        content='Tool call: external.web_search{"query":"C罗 2024年10月 比赛 schedule","max_results":5}'
                    )
                return LLMChatCompletion(content="C 罗本周赛程以官方公布为准。")

        result = ToolChoiceLoopRunner(registry=registry, llm_client=FakeLLM()).run(
            LoopAgentTask(
                user_message="你看一下c罗这个星期有什么比赛吗",
                available_capabilities=("external.web_search",),
                source_type="agent_chat",
            ),
            max_steps=2,
        )

        self.assertEqual(LoopAgentStopReason.MODEL_FINAL, result.stop_reason)
        self.assertEqual(1, len(executed_inputs))
        self.assertEqual(5, executed_inputs[0]["max_results"])
        self.assertIn("C罗", executed_inputs[0]["query"])
        self.assertIn("this week", executed_inputs[0]["query"])
        self.assertIn("Cristiano Ronaldo", executed_inputs[0]["query"])
        self.assertIn("Al Nassr", executed_inputs[0]["query"])
        self.assertIn("football fixtures", executed_inputs[0]["query"])
        self.assertIn(date.today().isoformat(), executed_inputs[0]["query"])
        self.assertNotIn("你看一下", executed_inputs[0]["query"])
        self.assertNotIn("2024年10月", executed_inputs[0]["query"])
        self.assertEqual("external.web_search", result.trace[0].capability)
        self.assertEqual("succeeded", result.trace[0].observation_status)

    def test_runner_rejects_missing_required_tool_argument_before_handler_runs(self) -> None:
        from app.agent_runtime.loop_agent.tool_choice_runner import LoopAgentTask, ToolChoiceLoopRunner
        from app.agent_runtime.loop_agent.schemas import LoopAgentStopReason
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        executed = []
        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="external.web_search",
                    description="查询公开网页资料。",
                    input_schema={
                        "type": "object",
                        "required": ["query"],
                        "properties": {"query": {"type": "string"}},
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=lambda _session, **arguments: executed.append(arguments) or {"ok": True},
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            ]
        )

        class FakeLLM:
            def complete(self, *, messages, tools=None, tool_choice=None):
                return LLMChatCompletion(
                    content="",
                    tool_calls=[LLMToolCall(id="call-missing-query", name="external_web_search", arguments={})],
                )

        result = ToolChoiceLoopRunner(registry=registry, llm_client=FakeLLM()).run(
            LoopAgentTask(
                user_message="查腾讯校招",
                available_capabilities=("external.web_search",),
                source_type="agent_chat",
            ),
            max_steps=2,
        )

        self.assertEqual(LoopAgentStopReason.STEP_FAILED, result.stop_reason)
        self.assertEqual([], executed)
        self.assertEqual("failed", result.trace[0].observation_status)
        self.assertEqual("TOOL_INPUT_INVALID", result.trace[0].metadata["observation"]["metadata"]["error_code"])


if __name__ == "__main__":
    unittest.main()
