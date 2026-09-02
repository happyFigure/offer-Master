import sys
import unittest
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class LoopAgentToolChoiceRunnerTest(unittest.TestCase):
    def test_runner_prompt_prefers_exact_replace_tool_for_preserve_file_edits(self) -> None:
        from app.agent_runtime.loop_agent.tool_choice_runner import LoopAgentTask, ToolChoiceLoopRunner
        from app.agent_runtime.tool_registry import AgentToolRegistry
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        class FakeLLM:
            def __init__(self) -> None:
                self.calls = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
                return LLMChatCompletion(content="不需要工具。")

        llm = FakeLLM()

        ToolChoiceLoopRunner(registry=AgentToolRegistry(), llm_client=llm).run(
            LoopAgentTask(user_message="把文件里的旧名字换成新名字，其他不要动"),
            max_steps=1,
        )

        system_prompt = str(llm.calls[0]["messages"][0]["content"])
        self.assertIn("精确替换", system_prompt)
        self.assertIn("不要让模型重写整个文件", system_prompt)

    def test_runner_advances_stage_context_after_successful_tool_observation(self) -> None:
        from app.agent_runtime.loop_agent.tool_choice_runner import LoopAgentTask, ToolChoiceLoopRunner
        from app.agent_runtime.loop_agent.schemas import LoopAgentStopReason
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        def fake_local_overview(_session, *, limit: int = 20):
            return {
                "ok": True,
                "result": {"summary": "正式企业表 3 家，岗位线索 71 条。"},
            }

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="local.company_database_overview",
                    description="查看本地公司数据库概览。",
                    input_schema={
                        "type": "object",
                        "properties": {"limit": {"type": "integer"}},
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=fake_local_overview,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            ]
        )

        class FakeLLM:
            def __init__(self) -> None:
                self.joined_prompts = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                joined_prompt = "\n".join(str(message.get("content") or "") for message in messages)
                self.joined_prompts.append(joined_prompt)
                if len(self.joined_prompts) == 1:
                    test_case.assertIn("当前任务阶段：明确目标和约束", joined_prompt)
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-local-overview",
                                name="local_company_database_overview",
                                arguments={"limit": 20},
                            )
                        ],
                    )
                test_case.assertIn("当前任务阶段：收集本地候选信息", messages[-1]["content"])
                test_case.assertIn("阶段标识：collect_candidates", messages[-1]["content"])
                return LLMChatCompletion(content="我已经进入收集本地候选信息阶段。")

        test_case = self
        llm = FakeLLM()
        result = ToolChoiceLoopRunner(registry=registry, llm_client=llm).run(
            LoopAgentTask(
                user_message="帮我分析数据库里的公司机会",
                available_capabilities=("local.company_database_overview",),
                source_type="agent_chat",
                stage_context={
                    "stage_id": "clarify_goal",
                    "title": "明确目标和约束",
                    "objective": "确认用户要完成什么。",
                    "status": "running",
                    "stage_plan": [
                        {
                            "stage_id": "clarify_goal",
                            "title": "明确目标和约束",
                            "objective": "确认用户要完成什么。",
                            "status": "running",
                        },
                        {
                            "stage_id": "collect_candidates",
                            "title": "收集本地候选信息",
                            "objective": "收集本地数据库里的候选公司和岗位线索。",
                            "status": "pending",
                        },
                    ],
                },
            ),
            max_steps=2,
        )

        self.assertEqual(LoopAgentStopReason.MODEL_FINAL, result.stop_reason)
        self.assertEqual(2, len(llm.joined_prompts))
        self.assertEqual("collect_candidates", result.metadata["stage_context"]["stage_id"])
        self.assertEqual(["clarify_goal", "collect_candidates"], result.metadata["stage_context_history"])

    def test_runner_filters_offered_tools_by_active_stage_strategy(self) -> None:
        from app.agent_runtime.loop_agent.tool_choice_runner import LoopAgentTask, ToolChoiceLoopRunner
        from app.agent_runtime.loop_agent.schemas import LoopAgentStopReason
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="local.company_database_overview",
                    description="查看本地公司数据库概览。",
                    input_schema={"type": "object", "additionalProperties": False},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"ok": True},
                    allowed_source_types=frozenset({"agent_chat"}),
                ),
                AgentToolDefinition(
                    name="external.web_search",
                    description="查询公开网页资料。",
                    input_schema={"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"ok": True},
                    allowed_source_types=frozenset({"agent_chat"}),
                ),
            ]
        )

        class FakeLLM:
            def complete(self, *, messages, tools=None, tool_choice=None):
                tool_names = [tool["function"]["name"] for tool in tools or []]
                test_case.assertEqual(["local_company_database_overview"], tool_names)
                test_case.assertEqual("auto", tool_choice)
                return LLMChatCompletion(content="我会先看本地公司库。")

        test_case = self
        result = ToolChoiceLoopRunner(registry=registry, llm_client=FakeLLM()).run(
            LoopAgentTask(
                user_message="帮我分析数据库里的公司机会",
                available_capabilities=("local.company_database_overview", "external.web_search"),
                source_type="agent_chat",
                stage_context={
                    "stage_id": "collect_candidates",
                    "title": "收集本地候选信息",
                    "objective": "查询本地公司库、岗位线索、校招来源等已有信息。",
                    "status": "running",
                    "allowed_capabilities": ["local.company_database_overview"],
                },
            ),
            max_steps=1,
        )

        self.assertEqual(LoopAgentStopReason.MODEL_FINAL, result.stop_reason)
        self.assertEqual("collect_candidates", result.metadata["stage_context"]["stage_id"])

    def test_runner_switches_tool_strategy_after_stage_advance(self) -> None:
        from app.agent_runtime.loop_agent.tool_choice_runner import LoopAgentTask, ToolChoiceLoopRunner
        from app.agent_runtime.loop_agent.schemas import LoopAgentStopReason
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        def fake_local_overview(_session, **_arguments):
            return {"ok": True, "result": {"summary": "本地候选公司 2 家。"}}

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="local.company_database_overview",
                    description="查看本地公司数据库概览。",
                    input_schema={"type": "object", "additionalProperties": False},
                    output_schema={"type": "object"},
                    handler=fake_local_overview,
                    allowed_source_types=frozenset({"agent_chat"}),
                ),
                AgentToolDefinition(
                    name="external.web_search",
                    description="查询公开网页资料。",
                    input_schema={"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"ok": True},
                    allowed_source_types=frozenset({"agent_chat"}),
                ),
            ]
        )

        class FakeLLM:
            def __init__(self) -> None:
                self.tool_names_by_call: list[list[str]] = []

            def complete(self, *, messages, tools=None, tool_choice=None):
                tool_names = [tool["function"]["name"] for tool in tools or []]
                self.tool_names_by_call.append(tool_names)
                if len(self.tool_names_by_call) == 1:
                    test_case.assertEqual(["local_company_database_overview"], tool_names)
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-local-overview",
                                name="local_company_database_overview",
                                arguments={},
                            )
                        ],
                    )
                test_case.assertEqual(["external_web_search"], tool_names)
                test_case.assertIn("阶段标识：enrich_external_info", messages[-1]["content"])
                return LLMChatCompletion(content="接下来只应该补充外部公开信息。")

        test_case = self
        llm = FakeLLM()
        result = ToolChoiceLoopRunner(registry=registry, llm_client=llm).run(
            LoopAgentTask(
                user_message="帮我分析数据库里的公司机会",
                available_capabilities=("local.company_database_overview", "external.web_search"),
                source_type="agent_chat",
                stage_context={
                    "stage_id": "collect_candidates",
                    "title": "收集本地候选信息",
                    "objective": "查询本地已有公司和岗位线索。",
                    "status": "running",
                    "allowed_capabilities": ["local.company_database_overview"],
                    "stage_plan": [
                        {
                            "stage_id": "collect_candidates",
                            "title": "收集本地候选信息",
                            "objective": "查询本地已有公司和岗位线索。",
                            "status": "running",
                            "allowed_capabilities": ["local.company_database_overview"],
                        },
                        {
                            "stage_id": "enrich_external_info",
                            "title": "补充外部公开信息",
                            "objective": "联网补充候选公司的公开资料。",
                            "status": "pending",
                            "allowed_capabilities": ["external.web_search"],
                        },
                    ],
                },
            ),
            max_steps=2,
        )

        self.assertEqual(LoopAgentStopReason.MODEL_FINAL, result.stop_reason)
        self.assertEqual([["local_company_database_overview"], ["external_web_search"]], llm.tool_names_by_call)
        self.assertEqual(["collect_candidates", "enrich_external_info"], result.metadata["stage_context_history"])

    def test_runner_includes_stage_context_and_handoff_in_model_prompt(self) -> None:
        from app.agent_runtime.loop_agent.tool_choice_runner import (
            LoopAgentStageContext,
            LoopAgentTask,
            ToolChoiceLoopRunner,
        )
        from app.agent_runtime.loop_agent.schemas import LoopAgentStopReason
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="external.web_search",
                    description="查询公开网页资料。",
                    input_schema={"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"ok": True},
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            ]
        )

        class FakeLLM:
            def __init__(self) -> None:
                self.messages = None

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.messages = messages
                joined_prompt = "\n".join(str(message.get("content") or "") for message in messages)
                test_case.assertIn("当前任务阶段：补充外部公开信息", joined_prompt)
                test_case.assertIn("阶段标识：enrich_external_info", joined_prompt)
                test_case.assertIn("阶段目标：补充候选公司的公开资料", joined_prompt)
                test_case.assertIn("上游阶段交接信息", joined_prompt)
                test_case.assertIn("正式企业表 1 家", joined_prompt)
                test_case.assertIn("local.company_database_overview", joined_prompt)
                return LLMChatCompletion(content="我会基于上游候选信息补充公开资料。")

        test_case = self
        llm = FakeLLM()
        result = ToolChoiceLoopRunner(registry=registry, llm_client=llm).run(
            LoopAgentTask(
                user_message="帮我分析数据库里的公司机会",
                available_capabilities=("external.web_search",),
                source_type="agent_chat",
                stage_context=LoopAgentStageContext(
                    stage_id="enrich_external_info",
                    title="补充外部公开信息",
                    objective="补充候选公司的公开资料",
                    status="running",
                    received_context={
                        "summary": "正式企业表 1 家，岗位线索 1 条。",
                        "tool_names": ["local.company_database_overview"],
                        "upstream_stage_ids": ["collect_candidates"],
                    },
                ),
            ),
            max_steps=1,
        )

        self.assertEqual(LoopAgentStopReason.MODEL_FINAL, result.stop_reason)
        self.assertEqual("enrich_external_info", result.metadata["stage_context"]["stage_id"])

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

    def test_runner_converts_multiline_textual_tool_call_into_real_tool_execution(self) -> None:
        from app.agent_runtime.loop_agent.tool_choice_runner import LoopAgentTask, ToolChoiceLoopRunner
        from app.agent_runtime.loop_agent.schemas import LoopAgentStopReason
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        executed_inputs = []

        def fake_read_file(_session, *, path: str, encoding: str = "utf-8"):
            executed_inputs.append({"path": path, "encoding": encoding})
            return {
                "tool_name": "filesystem.read_file",
                "ok": True,
                "result": {"content": "简历姓名：刘汉卿"},
                "summary": "已读取文件。",
            }

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="filesystem.read_file",
                    description="读取用户指定的本地文件。",
                    input_schema={
                        "type": "object",
                        "required": ["path"],
                        "properties": {
                            "path": {"type": "string"},
                            "encoding": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=fake_read_file,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            ]
        )

        class FakeLLM:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls += 1
                if self.calls == 1:
                    return LLMChatCompletion(
                        content=(
                            "Tool call: filesystem.read_file\n"
                            'Arguments: {"path":"C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex","encoding":"utf-8"}'
                        )
                    )
                return LLMChatCompletion(content="我已经读取到文件，里面有：简历姓名：刘汉卿")

        result = ToolChoiceLoopRunner(registry=registry, llm_client=FakeLLM()).run(
            LoopAgentTask(
                user_message="请用 filesystem.read_file 读取路径 C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex",
                available_capabilities=("filesystem.read_file",),
                source_type="agent_chat",
            ),
            max_steps=2,
        )

        self.assertEqual(LoopAgentStopReason.MODEL_FINAL, result.stop_reason)
        self.assertEqual(
            [
                {
                    "path": "C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex",
                    "encoding": "utf-8",
                }
            ],
            executed_inputs,
        )
        self.assertEqual("filesystem.read_file", result.trace[0].capability)
        self.assertEqual("succeeded", result.trace[0].observation_status)
        self.assertNotIn("Tool call:", result.final_answer or "")

    def test_runner_prefers_arguments_line_when_textual_tool_call_has_explanation_text(self) -> None:
        from app.agent_runtime.loop_agent.tool_choice_runner import LoopAgentTask, ToolChoiceLoopRunner
        from app.agent_runtime.loop_agent.schemas import LoopAgentStopReason
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.infrastructure.llm.chat_client import LLMChatCompletion

        executed_inputs = []

        def fake_read_file(_session, *, path: str, encoding: str = "utf-8"):
            executed_inputs.append({"path": path, "encoding": encoding})
            return {"tool_name": "filesystem.read_file", "ok": True, "result": {"content": "姓名：刘汉卿"}}

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="filesystem.read_file",
                    description="读取用户指定的本地文件。",
                    input_schema={
                        "type": "object",
                        "required": ["path"],
                        "properties": {"path": {"type": "string"}, "encoding": {"type": "string"}},
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=fake_read_file,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            ]
        )

        class FakeLLM:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls += 1
                if self.calls == 1:
                    return LLMChatCompletion(
                        content=(
                            "Tool call: filesystem.read_file\n"
                            'Arguments: {"path":"C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex","encoding":"utf-8"}\n\n'
                            "我会先读取文件，再给你解释。"
                        )
                    )
                return LLMChatCompletion(content="读取完成。")

        result = ToolChoiceLoopRunner(registry=registry, llm_client=FakeLLM()).run(
            LoopAgentTask(
                user_message="读一下里面的内容",
                available_capabilities=("filesystem.read_file",),
                source_type="agent_chat",
            ),
            max_steps=2,
        )

        self.assertEqual(LoopAgentStopReason.MODEL_FINAL, result.stop_reason)
        self.assertEqual(
            [{"path": "C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex", "encoding": "utf-8"}],
            executed_inputs,
        )
        self.assertEqual("filesystem.read_file", result.trace[0].capability)
        self.assertEqual("succeeded", result.trace[0].observation_status)

    def test_runner_retries_filesystem_read_when_content_looks_garbled(self) -> None:
        from app.agent_runtime.loop_agent.tool_choice_runner import LoopAgentTask, ToolChoiceLoopRunner
        from app.agent_runtime.loop_agent.schemas import LoopAgentStopReason
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry
        from app.infrastructure.llm.chat_client import LLMChatCompletion, LLMToolCall

        executed_inputs = []

        def fake_read_file(_session, *, path: str, encoding: str = "auto"):
            executed_inputs.append({"path": path, "encoding": encoding})
            if encoding == "auto":
                return {
                    "tool_name": "filesystem.read_file",
                    "ok": True,
                    "result": {"content": "\\resumeHeader\n  {������}\n  {���� / AI Agent ��̨�з�}"},
                    "summary": "已读取文件。",
                }
            return {
                "tool_name": "filesystem.read_file",
                "ok": True,
                "result": {"content": "\\resumeHeader\n  {刘汉卿}\n  {后端开发 / AI Agent 平台研发}"},
                "summary": "已读取文件。",
            }

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="filesystem.read_file",
                    description="读取用户指定的本地文件。",
                    input_schema={
                        "type": "object",
                        "required": ["path"],
                        "properties": {"path": {"type": "string"}, "encoding": {"type": "string"}},
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object"},
                    handler=fake_read_file,
                    allowed_source_types=frozenset({"agent_chat"}),
                )
            ]
        )

        class FakeLLM:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, *, messages, tools=None, tool_choice=None):
                self.calls += 1
                if self.calls == 1:
                    return LLMChatCompletion(
                        content="",
                        tool_calls=[
                            LLMToolCall(
                                id="call-read-garbled",
                                name="filesystem_read_file",
                                arguments={"path": "C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex", "encoding": "auto"},
                            )
                        ],
                    )
                return LLMChatCompletion(content="已读取到正常内容，姓名是刘汉卿。")

        result = ToolChoiceLoopRunner(registry=registry, llm_client=FakeLLM()).run(
            LoopAgentTask(
                user_message="读取这个简历文件内容",
                available_capabilities=("filesystem.read_file",),
                source_type="agent_chat",
            ),
            max_steps=3,
        )

        self.assertEqual(LoopAgentStopReason.MODEL_FINAL, result.stop_reason)
        self.assertEqual(
            [
                {"path": "C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex", "encoding": "auto"},
                {"path": "C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex", "encoding": "utf-8"},
            ],
            executed_inputs,
        )
        self.assertEqual("partial", result.trace[0].observation_status)
        self.assertIn("乱码", result.trace[0].observation_summary or "")
        observation_metadata = result.trace[0].metadata["observation"]["metadata"]
        self.assertEqual("TOOL_RESULT_GARBLED", observation_metadata["quality_gate"])
        self.assertEqual("encoding_quality", result.trace[0].metadata["observation"]["suggested_next_decision"]["metadata"]["source"])
        self.assertEqual("succeeded", result.trace[1].observation_status)

    def test_runner_waits_for_user_when_required_tool_argument_is_missing(self) -> None:
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

        self.assertEqual(LoopAgentStopReason.WAITING_USER, result.stop_reason)
        self.assertTrue(result.requires_user_action)
        self.assertEqual([], executed)
        self.assertEqual("waiting_user", result.trace[0].observation_status)
        self.assertIn("缺少 query", result.trace[0].observation_summary)
        self.assertEqual("TOOL_INPUT_INVALID", result.trace[0].metadata["observation"]["metadata"]["error_code"])


if __name__ == "__main__":
    unittest.main()
