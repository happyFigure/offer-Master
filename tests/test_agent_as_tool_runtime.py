import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class AgentAsToolRuntimeTest(unittest.TestCase):
    def test_capability_definition_can_be_created_from_existing_tool_definition(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentCapabilityDefinition
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRiskLevel

        capability = AgentCapabilityDefinition.from_tool_definition(
            AgentToolDefinition(
                name="external.web_search",
                description="搜索公开网页招聘信息。",
                input_schema={"type": "object", "required": ["query"]},
                output_schema={"type": "object", "required": ["result"]},
                risk_level=AgentToolRiskLevel.LOW,
                requires_confirmation=False,
                allowed_source_types=frozenset({"agent_chat"}),
            ),
            executor_id="claude-sdk-agent",
            supported_intents=("campus_recruiting_search",),
        )

        self.assertEqual("external.web_search", capability.capability_id)
        self.assertEqual("external.web_search", capability.name)
        self.assertEqual("搜索公开网页招聘信息。", capability.description)
        self.assertEqual("claude-sdk-agent", capability.executor_id)
        self.assertEqual({"type": "object", "required": ["query"]}, capability.input_schema)
        self.assertEqual({"type": "object", "required": ["result"]}, capability.output_schema)
        self.assertEqual("low", capability.risk_level)
        self.assertEqual(("campus_recruiting_search",), capability.supported_intents)
        self.assertFalse(capability.requires_confirmation)
        self.assertEqual(frozenset({"agent_chat"}), capability.allowed_source_types)

    def test_registry_rejects_duplicate_capability_ids(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentCapabilityDefinition, AgentCapabilityRegistry

        definition = AgentCapabilityDefinition(
            capability_id="recruiting.search",
            name="校招网页搜索",
            description="搜索公开校招信息和官网入口。",
            executor_id="claude-sdk-agent",
            input_schema={"type": "object", "required": ["company_name"]},
            output_schema={"type": "object"},
            risk_level="low",
            supported_intents=("campus_recruiting_search",),
        )

        registry = AgentCapabilityRegistry([definition])

        with self.assertRaisesRegex(ValueError, "Agent capability already registered"):
            registry.register(definition)

    def test_registry_can_be_built_from_existing_tool_registry(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentCapabilityRegistry
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry

        tool_registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="external.web_search",
                    description="搜索公开网页招聘信息。",
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object"},
                ),
                AgentToolDefinition(
                    name="applications.find_apply_entry",
                    description="寻找申请入口。",
                    input_schema={"type": "object", "required": ["job_id"]},
                    output_schema={"type": "object"},
                    requires_confirmation=True,
                ),
            ]
        )

        registry = AgentCapabilityRegistry.from_tool_registry(
            tool_registry,
            default_executor_id="local-tool-agent",
            executor_id_by_capability={"external.web_search": "claude-sdk-agent"},
            supported_intents_by_capability={"external.web_search": ("campus_recruiting_search",)},
        )

        definitions = registry.list_definitions()
        self.assertEqual(["applications.find_apply_entry", "external.web_search"], [item.capability_id for item in definitions])
        self.assertEqual("claude-sdk-agent", registry.get("external.web_search").executor_id)
        self.assertEqual(("campus_recruiting_search",), registry.get("external.web_search").supported_intents)
        self.assertEqual("local-tool-agent", registry.get("applications.find_apply_entry").executor_id)
        self.assertTrue(registry.get("applications.find_apply_entry").requires_confirmation)

    def test_default_capability_registry_registers_current_tools_with_intents(self) -> None:
        from app.agent_runtime.agent_as_tool import TOOL_REGISTRY_EXECUTOR_ID, create_default_agent_capability_registry
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, OFFERIO_COMPANY_JOBS_TOOL

        registry = create_default_agent_capability_registry()

        web_search = registry.get(EXTERNAL_WEB_SEARCH_TOOL)
        offerio_sync = registry.get(OFFERIO_COMPANY_JOBS_TOOL)

        self.assertIsNotNone(web_search)
        self.assertIsNotNone(offerio_sync)
        self.assertEqual(TOOL_REGISTRY_EXECUTOR_ID, web_search.executor_id)
        self.assertEqual(("campus_recruiting_search", "external_agent_task"), web_search.supported_intents)
        self.assertEqual(("offerio_company_jobs_sync",), offerio_sync.supported_intents)

    def test_default_tool_registry_registers_filesystem_skill_tools(self) -> None:
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry

        registry = create_default_agent_tool_registry()

        self.assertIsNotNone(registry.get("filesystem.read_file"))
        self.assertIsNotNone(registry.get("filesystem.write_text"))
        self.assertIsNotNone(registry.get("filesystem.replace_text"))
        self.assertIsNotNone(registry.get("filesystem.delete_path"))
        self.assertFalse(registry.get("filesystem.read_file").requires_confirmation)
        self.assertTrue(registry.get("filesystem.write_text").requires_confirmation)
        self.assertTrue(registry.get("filesystem.replace_text").requires_confirmation)
        self.assertTrue(registry.get("filesystem.delete_path").requires_confirmation)

    def test_filesystem_read_file_tool_runs_packaged_script(self) -> None:
        from app.agent_runtime.tool_registry import create_filesystem_agent_tool_definitions

        skill_root = PROJECT_ROOT / "runtime" / "test-temp" / "filesystem-skill-root"
        scripts_dir = skill_root / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script_file = scripts_dir / "read_file.py"
        script_file.write_text(
            "import argparse\n"
            "from pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--path', required=True)\n"
            "parser.add_argument('--offset', default='0')\n"
            "parser.add_argument('--limit', default='0')\n"
            "parser.add_argument('--encoding', default='utf-8')\n"
            "args = parser.parse_args()\n"
            "print(Path(args.path).read_text(encoding=args.encoding))\n",
            encoding="utf-8",
        )
        source_file = skill_root / "resume.tex"
        source_file.write_text("line one\n姓名：刘汉卿\nline two", encoding="utf-8")

        read_file = next(
            definition
            for definition in create_filesystem_agent_tool_definitions(script_root=skill_root)
            if definition.name == "filesystem.read_file"
        )

        result = read_file.handler(None, path=str(source_file), encoding="utf-8", offset=0, limit=1)

        self.assertTrue(result["ok"])
        self.assertEqual("filesystem.read_file", result["tool_name"])
        self.assertIn("line one", result["result"]["content"])
        self.assertIn("姓名：刘汉卿", result["result"]["content"])
        self.assertNotIn("�", result["result"]["content"])
        self.assertEqual(0, result["result"]["return_code"])

    def test_filesystem_write_tool_requires_runtime_confirmation(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolNextAction, AgentToolRuntimeGuard
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry

        result = AgentToolRuntimeGuard().pre_check(
            AgentToolCallContext(
                stage="maybe_tool",
                tool_name="filesystem.write_text",
                source_type="agent_chat",
                user_confirmed=False,
            ),
            registry=create_default_agent_tool_registry(),
        )

        self.assertFalse(result.ok)
        self.assertEqual(AgentToolNextAction.REQUEST_USER_CONFIRMATION.value, result.next_action)

    def test_filesystem_replace_text_tool_runs_packaged_script(self) -> None:
        from app.agent_runtime.tool_registry import create_filesystem_agent_tool_definitions

        skill_root = PROJECT_ROOT / "docs" / "agent-skills" / "0539e315-2960-45bf-ae7c-1a7abc4e6755"
        source_file = PROJECT_ROOT / "runtime" / "test-temp" / "replace-text-resume.tex"
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text("姓名：刘汉卿\n项目：保持原样", encoding="utf-8")

        replace_text = next(
            definition
            for definition in create_filesystem_agent_tool_definitions(script_root=skill_root)
            if definition.name == "filesystem.replace_text"
        )

        result = replace_text.handler(
            None,
            path=str(source_file),
            old_text="刘汉卿",
            new_text="王爷",
            encoding="utf-8",
            count=0,
        )

        self.assertTrue(result["ok"])
        self.assertEqual("filesystem.replace_text", result["tool_name"])
        self.assertEqual("姓名：王爷\n项目：保持原样", source_file.read_text(encoding="utf-8"))
        self.assertIn("REPLACED", result["result"]["stdout"])

    def test_tool_registry_executor_runs_registered_handler_through_runtime(self) -> None:
        from app.agent_runtime.agent_as_tool import (
            TOOL_REGISTRY_EXECUTOR_ID,
            AgentCapabilityRegistry,
            AgentRuntime,
            AgentRuntimeContext,
            AgentTask,
            ToolRegistryAgentExecutor,
        )
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry

        calls = []

        def sync_handler(session, **arguments):
            calls.append((session, arguments))
            return {
                "tool_name": "offerio.sync_company_jobs",
                "ok": True,
                "result": {"status": "succeeded", "synced_count": arguments["limit"]},
            }

        tool_registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="offerio.sync_company_jobs",
                    description="同步 OfferIO 公司聚合岗位库。",
                    input_schema={"type": "object", "required": ["limit"], "properties": {"limit": {"type": "integer"}}},
                    output_schema={"type": "object"},
                    handler=sync_handler,
                )
            ]
        )
        capability_registry = AgentCapabilityRegistry.from_tool_registry(
            tool_registry,
            default_executor_id=TOOL_REGISTRY_EXECUTOR_ID,
        )
        runtime = AgentRuntime(
            registry=capability_registry,
            executors={
                TOOL_REGISTRY_EXECUTOR_ID: ToolRegistryAgentExecutor(
                    tool_registry,
                    session_provider=lambda context: f"db-session-for-{context.session_id}",
                )
            },
        )

        result = runtime.call(
            AgentTask(
                capability_id="offerio.sync_company_jobs",
                goal="同步 OfferIO 公司聚合岗位库",
                input_payload={"limit": 25},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="task-1"),
        )

        self.assertEqual("succeeded", result.status)
        self.assertEqual("offerio.sync_company_jobs 执行成功", result.summary)
        self.assertEqual({"status": "succeeded", "synced_count": 25}, result.raw_result["result"])
        self.assertEqual([("db-session-for-session-1", {"limit": 25})], calls)

    def test_tool_registry_executor_standardizes_failed_handler_payload(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentTask, AgentRuntimeContext, ToolRegistryAgentExecutor
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry

        tool_registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="external.web_search",
                    description="搜索公开网页招聘信息。",
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {
                        "tool_name": "external.web_search",
                        "ok": False,
                        "error": "EXTERNAL_WEB_SEARCH_NOT_CONFIGURED",
                    },
                )
            ]
        )

        result = ToolRegistryAgentExecutor(tool_registry).call(
            AgentTask(
                capability_id="external.web_search",
                goal="查腾讯校招",
                input_payload={"query": "腾讯 校招"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="task-1"),
        )

        self.assertEqual("failed", result.status)
        self.assertEqual("external.web_search 执行失败：EXTERNAL_WEB_SEARCH_NOT_CONFIGURED", result.summary)
        self.assertEqual({"tool_name": "external.web_search", "ok": False, "error": "EXTERNAL_WEB_SEARCH_NOT_CONFIGURED"}, result.raw_result)

    def test_claude_sdk_agent_executor_runs_web_search_as_direct_agent_tool(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.external_tasks.executors import ClaudeSdkAgentExecutor
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL

        calls = []

        class FakeClaudeSdkAdapter:
            executor_name = "claude-sdk-agent"

            def execute_web_search(self, query: str, *, max_results: int = 5):
                calls.append({"query": query, "max_results": max_results})
                return {
                    "executor_name": "claude-sdk-agent",
                    "query": query,
                    "answer": "腾讯校招官网：https://join.qq.com/",
                    "observations": ["腾讯校招官网可查看产品岗。"],
                    "artifacts": [{"type": "url", "title": "腾讯校招", "url": "https://join.qq.com/"}],
                }

        result = ClaudeSdkAgentExecutor(FakeClaudeSdkAdapter()).call(
            AgentTask(
                capability_id=EXTERNAL_WEB_SEARCH_TOOL,
                goal="查腾讯校招官网",
                input_payload={"query": "腾讯 校园招聘 官网", "max_results": 3},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="task-1"),
        )

        self.assertEqual([{"query": "腾讯 校园招聘 官网", "max_results": 3}], calls)
        self.assertEqual("succeeded", result.status)
        self.assertEqual("腾讯校招官网：https://join.qq.com/", result.summary)
        self.assertEqual("腾讯校招官网可查看产品岗。", result.observation)
        self.assertEqual([{"type": "url", "title": "腾讯校招", "url": "https://join.qq.com/"}], result.evidence)
        self.assertEqual(EXTERNAL_WEB_SEARCH_TOOL, result.raw_result["tool_name"])
        self.assertTrue(result.raw_result["ok"])
        self.assertEqual("claude-sdk-agent", result.raw_result["result"]["executor_name"])

    def test_claude_sdk_agent_executor_declares_web_search_capability(self) -> None:
        from app.agent_runtime.agent_as_tool import CLAUDE_SDK_AGENT_EXECUTOR_ID
        from app.agent_runtime.external_tasks.executors import ClaudeSdkAgentExecutor
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL

        class FakeClaudeSdkAdapter:
            executor_name = "claude-sdk-agent"

            def execute_find_apply_entry(self, envelope):
                raise AssertionError("not needed")

            def execute_web_search(self, query: str, *, max_results: int = 5):
                raise AssertionError("not needed")

        capabilities = ClaudeSdkAgentExecutor(FakeClaudeSdkAdapter()).capabilities()

        self.assertEqual([EXTERNAL_WEB_SEARCH_TOOL], [definition.capability_id for definition in capabilities])
        capability = capabilities[0]
        self.assertEqual(CLAUDE_SDK_AGENT_EXECUTOR_ID, capability.executor_id)
        self.assertEqual(("campus_recruiting_search", "external_agent_task"), capability.supported_intents)
        self.assertEqual(["query"], capability.input_schema["required"])

    def test_openai_sdk_agent_executor_declares_resume_tailoring_capability(self) -> None:
        from app.agent_runtime.agent_as_tool import OPENAI_SDK_AGENT_EXECUTOR_ID
        from app.agent_runtime.external_tasks.executors import OpenAISdkAgentExecutor

        class FakeOpenAIResumeClient:
            executor_name = "openai-sdk-agent"

            def execute_resume_tailoring(self, **_arguments):
                raise AssertionError("not needed")

        capabilities = OpenAISdkAgentExecutor(FakeOpenAIResumeClient()).capabilities()

        self.assertEqual(["resume.tailor"], [definition.capability_id for definition in capabilities])
        capability = capabilities[0]
        self.assertEqual(OPENAI_SDK_AGENT_EXECUTOR_ID, capability.executor_id)
        self.assertEqual("简历修改", capability.name)
        self.assertEqual(["resume_text", "job_description"], capability.input_schema["required"])
        self.assertEqual(("resume_tailoring",), capability.supported_intents)
        self.assertEqual("low", capability.risk_level)
        self.assertFalse(capability.requires_confirmation)
        self.assertEqual(frozenset({"agent_chat"}), capability.allowed_source_types)
        self.assertIn("resume_tailoring", capability.candidate_profile.categories)

    def test_openai_sdk_agent_executor_runs_resume_tailoring(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext, AgentTask
        from app.agent_runtime.external_tasks.executors import OpenAISdkAgentExecutor

        calls = []

        class FakeOpenAIResumeClient:
            executor_name = "openai-sdk-agent"

            def execute_resume_tailoring(self, **arguments):
                calls.append(arguments)
                return {
                    "executor_name": "openai-sdk-agent",
                    "revised_resume": "改写后：突出 Spring Boot、接口性能优化和 MySQL 经验。",
                    "change_summary": ["根据 Java 后端 JD 强化项目关键词", "保留原始经历，不新增虚构内容"],
                    "warnings": [],
                }

        result = OpenAISdkAgentExecutor(FakeOpenAIResumeClient()).call(
            AgentTask(
                capability_id="resume.tailor",
                goal="根据 JD 修改简历",
                input_payload={
                    "resume_text": "做过 Java 后端项目，使用 Spring Boot 和 MySQL。",
                    "job_description": "Java 后端开发，要求 Spring Boot、数据库和接口性能优化。",
                    "language": "zh-CN",
                },
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="task-1"),
        )

        self.assertEqual("succeeded", result.status)
        self.assertEqual("OpenAI SDK agent 已完成简历修改", result.summary)
        self.assertIn("改写后", result.observation)
        self.assertEqual("resume.tailor", result.raw_result["tool_name"])
        self.assertTrue(result.raw_result["ok"])
        self.assertEqual("openai-sdk-agent", result.raw_result["result"]["executor_name"])
        self.assertEqual("做过 Java 后端项目，使用 Spring Boot 和 MySQL。", calls[0]["resume_text"])
        self.assertEqual("Java 后端开发，要求 Spring Boot、数据库和接口性能优化。", calls[0]["job_description"])

    def test_openai_sdk_resume_client_adapter_uses_chat_completion_json(self) -> None:
        from app.agent_runtime.external_tasks.executors import OpenAISdkAgentConfig, OpenAISdkResumeClientAdapter

        calls = []

        class FakeCompletions:
            def create(self, **payload):
                calls.append(payload)
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"revised_resume":"新版简历","change_summary":["突出 JD 关键词"],"warnings":[]}'
                            }
                        }
                    ]
                }

        class FakeChat:
            completions = FakeCompletions()

        class FakeOpenAIClient:
            chat = FakeChat()

        adapter = OpenAISdkResumeClientAdapter(
            config=OpenAISdkAgentConfig(model="gpt-test", timeout_seconds=12.0),
            client=FakeOpenAIClient(),
        )

        result = adapter.execute_resume_tailoring(
            resume_text="原简历：Java 项目",
            job_description="目标 JD：Java 后端",
            language="zh-CN",
        )

        self.assertEqual("新版简历", result["revised_resume"])
        self.assertEqual(["突出 JD 关键词"], result["change_summary"])
        self.assertEqual("gpt-test", calls[0]["model"])
        self.assertEqual(12.0, calls[0]["timeout"])
        self.assertEqual({"type": "json_object"}, calls[0]["response_format"])
        self.assertIn("原简历：Java 项目", calls[0]["messages"][1]["content"])
        self.assertIn("目标 JD：Java 后端", calls[0]["messages"][1]["content"])

    def test_runtime_bundle_collects_capabilities_declared_by_agents(self) -> None:
        from app.agent_runtime.agent_as_tool import (
            AgentCapabilityDefinition,
            AgentRuntime,
            AgentRuntimeContext,
            AgentTask,
            StandardAgentResult,
            build_agent_runtime_bundle,
        )

        class FakeOpenAIAgent:
            executor_id = "openai-sdk-agent"

            def capabilities(self):
                return [
                    AgentCapabilityDefinition(
                        capability_id="resume.tailor",
                        name="简历优化",
                        description="根据 JD 优化简历表达。",
                        executor_id=self.executor_id,
                        input_schema={"type": "object", "required": ["resume_text", "job_description"]},
                        output_schema={"type": "object", "required": ["revised_resume"]},
                        risk_level="medium",
                        supported_intents=("resume_tailoring",),
                    )
                ]

            def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
                return StandardAgentResult(
                    status="succeeded",
                    summary=f"{context.namespace} 完成简历优化",
                    raw_result={"revised_resume": "突出 Java 后端项目经验"},
                )

        bundle = build_agent_runtime_bundle([FakeOpenAIAgent()])

        self.assertEqual(["openai-sdk-agent"], sorted(bundle.executors))
        self.assertEqual("openai-sdk-agent", bundle.capability_executor_ids["resume.tailor"])
        self.assertEqual("openai-sdk-agent", bundle.capability_registry.get("resume.tailor").executor_id)
        self.assertEqual(("resume_tailoring",), bundle.capability_registry.get("resume.tailor").supported_intents)

        runtime = AgentRuntime(registry=bundle.capability_registry, executors=bundle.executors)
        result = runtime.call(
            AgentTask(
                capability_id="resume.tailor",
                goal="优化简历",
                input_payload={"resume_text": "old", "job_description": "java backend"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="task-1"),
        )

        self.assertEqual("succeeded", result.status)
        self.assertEqual("openai-sdk-agent 完成简历优化", result.summary)

    def test_declared_agent_capability_overrides_legacy_tool_registry_capability(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentCapabilityDefinition, build_agent_runtime_bundle
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry

        class FakeClaudeAgent:
            executor_id = "claude-sdk-agent"

            def capabilities(self):
                return [
                    AgentCapabilityDefinition(
                        capability_id=EXTERNAL_WEB_SEARCH_TOOL,
                        name="网页搜索",
                        description="由 Claude SDK agent 搜索公开网页。",
                        executor_id=self.executor_id,
                        input_schema={"type": "object", "required": ["query"]},
                        output_schema={"type": "object"},
                        risk_level="low",
                        supported_intents=("campus_recruiting_search",),
                    )
                ]

            def call(self, task, context):
                raise AssertionError("not needed for registration test")

        tool_registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="旧工具系统里的网页搜索。",
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object"},
                )
            ]
        )

        bundle = build_agent_runtime_bundle([FakeClaudeAgent()], tool_registry=tool_registry)

        capability = bundle.capability_registry.get(EXTERNAL_WEB_SEARCH_TOOL)
        self.assertEqual("claude-sdk-agent", capability.executor_id)
        self.assertEqual("由 Claude SDK agent 搜索公开网页。", capability.description)
        self.assertEqual("claude-sdk-agent", bundle.capability_executor_ids[EXTERNAL_WEB_SEARCH_TOOL])

    def test_runtime_calls_registered_agent_and_returns_standard_result(self) -> None:
        from app.agent_runtime.agent_as_tool import (
            AgentCapabilityDefinition,
            AgentCapabilityRegistry,
            AgentRuntime,
            AgentRuntimeContext,
            AgentTask,
            StandardAgentResult,
        )

        calls = []

        class FakeSearchAgent:
            def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
                calls.append((task, context))
                return StandardAgentResult(
                    status="succeeded",
                    summary="找到腾讯 2026 校招产品岗入口。",
                    observation="腾讯校招官网显示产品岗城市包括深圳、北京。",
                    evidence=[{"type": "url", "title": "腾讯校招", "url": "https://careers.tencent.com"}],
                    missing_information=["用户未指定城市偏好"],
                    next_actions=["建议调用简历匹配能力"],
                    raw_result={"source": "fake-search"},
                )

        registry = AgentCapabilityRegistry(
            [
                AgentCapabilityDefinition(
                    capability_id="recruiting.search",
                    name="校招网页搜索",
                    description="搜索公开校招信息和官网入口。",
                    executor_id="claude-sdk-agent",
                    input_schema={"type": "object", "required": ["company_name"]},
                    output_schema={"type": "object"},
                    risk_level="low",
                    supported_intents=("campus_recruiting_search",),
                )
            ]
        )
        runtime = AgentRuntime(registry=registry, executors={"claude-sdk-agent": FakeSearchAgent()})

        result = runtime.call(
            AgentTask(
                capability_id="recruiting.search",
                goal="查腾讯 2026 校招产品岗",
                input_payload={"company_name": "腾讯"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="task-1"),
        )

        self.assertEqual("succeeded", result.status)
        self.assertEqual("找到腾讯 2026 校招产品岗入口。", result.summary)
        self.assertEqual("腾讯校招官网显示产品岗城市包括深圳、北京。", result.observation)
        self.assertEqual(["用户未指定城市偏好"], result.missing_information)
        self.assertEqual("recruiting.search", calls[0][0].capability_id)
        self.assertEqual("session-1", calls[0][1].session_id)
        self.assertEqual("claude-sdk-agent", calls[0][1].namespace)

    def test_runtime_adds_result_envelope_for_any_agent_result(self) -> None:
        from app.agent_runtime.agent_as_tool import (
            AgentCapabilityDefinition,
            AgentCapabilityRegistry,
            AgentRuntime,
            AgentRuntimeContext,
            AgentTask,
            StandardAgentResult,
        )

        class FakeResumeAgent:
            def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
                return StandardAgentResult(
                    status="succeeded",
                    summary="简历已按 Java 后端岗位改写。",
                    observation="新版简历突出 Spring Boot 项目、接口性能优化和数据库经验。",
                    evidence=[{"type": "text", "title": "改写摘要", "url": "memory://resume-tailor/1"}],
                    missing_information=["目标城市偏好"],
                    next_actions=["继续调用岗位匹配能力"],
                    raw_result={"revised_resume": "突出 Spring Boot 和 MySQL 项目经验"},
                )

        runtime = AgentRuntime(
            registry=AgentCapabilityRegistry(
                [
                    AgentCapabilityDefinition(
                        capability_id="resume.tailor",
                        name="简历优化",
                        description="根据目标岗位优化简历表达。",
                        executor_id="openai-sdk-agent",
                        input_schema={"type": "object", "required": ["resume_text", "job_description"]},
                        output_schema={"type": "object", "required": ["revised_resume"]},
                        risk_level="medium",
                        supported_intents=("resume_tailoring",),
                    )
                ]
            ),
            executors={"openai-sdk-agent": FakeResumeAgent()},
        )

        result = runtime.call(
            AgentTask(
                capability_id="resume.tailor",
                goal="优化简历",
                input_payload={"resume_text": "old", "job_description": "java backend"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="task-1"),
        )

        self.assertEqual("succeeded", result.status)
        self.assertEqual("resume.tailor", result.raw_result["tool_name"])
        self.assertTrue(result.raw_result["ok"])
        self.assertEqual(["目标城市偏好"], result.raw_result["missing_information"])
        self.assertEqual(["继续调用岗位匹配能力"], result.raw_result["next_actions"])

        envelope = result.raw_result["result_envelope"]
        self.assertEqual("succeeded", envelope["status"])
        self.assertEqual("resume.tailor", envelope["capability"])
        self.assertEqual("openai-sdk-agent", envelope["executor"])
        self.assertEqual("简历已按 Java 后端岗位改写。", envelope["summary"])
        self.assertEqual("medium", envelope["risk_level"])
        self.assertEqual(
            ["新版简历突出 Spring Boot 项目、接口性能优化和数据库经验。"],
            envelope["observations"],
        )
        self.assertEqual([{"type": "text", "title": "改写摘要", "url": "memory://resume-tailor/1"}], envelope["artifacts"])
        self.assertEqual({"revised_resume": "突出 Spring Boot 和 MySQL 项目经验"}, envelope["raw_result"])

    def test_runtime_blocks_missing_required_input_before_calling_agent(self) -> None:
        from app.agent_runtime.agent_as_tool import (
            AgentCapabilityDefinition,
            AgentCapabilityRegistry,
            AgentRuntime,
            AgentRuntimeContext,
            AgentTask,
            StandardAgentResult,
        )

        class FailingAgent:
            def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
                raise AssertionError("agent should not be called when required input is missing")

        registry = AgentCapabilityRegistry(
            [
                AgentCapabilityDefinition(
                    capability_id="recruiting.search",
                    name="校招网页搜索",
                    description="搜索公开校招信息和官网入口。",
                    executor_id="claude-sdk-agent",
                    input_schema={"type": "object", "required": ["company_name"]},
                    output_schema={"type": "object"},
                    risk_level="low",
                    supported_intents=("campus_recruiting_search",),
                )
            ]
        )
        runtime = AgentRuntime(registry=registry, executors={"claude-sdk-agent": FailingAgent()})

        result = runtime.call(
            AgentTask(
                capability_id="recruiting.search",
                goal="查腾讯 2026 校招产品岗",
                input_payload={},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="task-1"),
        )

        self.assertEqual("failed", result.status)
        self.assertEqual("缺少必要输入：company_name", result.summary)
        self.assertEqual(["company_name"], result.missing_information)

    def test_runtime_requires_user_confirmation_before_calling_agent(self) -> None:
        from app.agent_runtime.agent_as_tool import (
            AgentCapabilityDefinition,
            AgentCapabilityRegistry,
            AgentRuntime,
            AgentRuntimeContext,
            AgentTask,
            StandardAgentResult,
        )

        class FailingAgent:
            def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
                raise AssertionError("agent should not be called before user confirmation")

        runtime = AgentRuntime(
            registry=AgentCapabilityRegistry(
                [
                    AgentCapabilityDefinition(
                        capability_id="applications.submit",
                        name="自动投递",
                        description="替用户提交岗位申请。",
                        executor_id="browser-agent",
                        input_schema={"type": "object", "required": ["job_id"]},
                        output_schema={"type": "object"},
                        risk_level="high",
                        requires_confirmation=True,
                    )
                ]
            ),
            executors={"browser-agent": FailingAgent()},
        )

        result = runtime.call(
            AgentTask(
                capability_id="applications.submit",
                goal="帮我投递这个岗位",
                input_payload={"job_id": "lead-1"},
            ),
            AgentRuntimeContext(
                session_id="session-1",
                run_id="run-1",
                task_id="task-1",
                permission_scope={"source_type": "agent_chat", "user_confirmed": False},
            ),
        )

        self.assertEqual("waiting_user", result.status)
        self.assertTrue(result.requires_user_action)
        self.assertEqual("能力需要用户确认后才能执行：applications.submit", result.summary)
        self.assertEqual("AGENT_CAPABILITY_USER_CONFIRMATION_REQUIRED", result.raw_result["permission"]["error_code"])
        self.assertFalse(result.raw_result["ok"])
        self.assertEqual("waiting_user", result.raw_result["result_envelope"]["status"])

    def test_runtime_blocks_disallowed_source_type_before_calling_agent(self) -> None:
        from app.agent_runtime.agent_as_tool import (
            AgentCapabilityDefinition,
            AgentCapabilityRegistry,
            AgentRuntime,
            AgentRuntimeContext,
            AgentTask,
            StandardAgentResult,
        )

        class FailingAgent:
            def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
                raise AssertionError("agent should not be called from a disallowed source type")

        runtime = AgentRuntime(
            registry=AgentCapabilityRegistry(
                [
                    AgentCapabilityDefinition(
                        capability_id="resume.tailor",
                        name="简历优化",
                        description="根据目标岗位优化简历表达。",
                        executor_id="openai-sdk-agent",
                        input_schema={"type": "object", "required": ["resume_text", "job_description"]},
                        output_schema={"type": "object"},
                        risk_level="medium",
                        allowed_source_types=frozenset({"agent_chat"}),
                    )
                ]
            ),
            executors={"openai-sdk-agent": FailingAgent()},
        )

        result = runtime.call(
            AgentTask(
                capability_id="resume.tailor",
                goal="优化简历",
                input_payload={"resume_text": "old", "job_description": "java backend"},
            ),
            AgentRuntimeContext(
                session_id="session-1",
                run_id="run-1",
                task_id="task-1",
                permission_scope={"source_type": "browser_agent"},
            ),
        )

        self.assertEqual("failed", result.status)
        self.assertEqual("能力不允许当前来源调用：resume.tailor", result.summary)
        self.assertEqual("AGENT_CAPABILITY_SOURCE_TYPE_NOT_ALLOWED", result.raw_result["permission"]["error_code"])
        self.assertEqual("browser_agent", result.raw_result["permission"]["source_type"])
        self.assertEqual(["agent_chat"], result.raw_result["permission"]["allowed_source_types"])
        self.assertFalse(result.raw_result["ok"])

    def test_runtime_retries_transient_sdk_failure_before_succeeding(self) -> None:
        from app.agent_runtime.agent_as_tool import (
            AgentCapabilityDefinition,
            AgentCapabilityRegistry,
            AgentRuntime,
            AgentRuntimeContext,
            AgentRuntimeRetryPolicy,
            AgentTask,
            StandardAgentResult,
        )

        calls = []

        class FlakySdkAgent:
            def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
                calls.append({"task": task, "context": context})
                if len(calls) < 3:
                    return StandardAgentResult(
                        status="failed",
                        summary="SDK 调用超时",
                        raw_result={
                            "tool_name": "external.web_search",
                            "ok": False,
                            "error": "request timed out",
                            "error_type": "APITimeoutError",
                        },
                    )
                return StandardAgentResult(
                    status="succeeded",
                    summary="腾讯校招官网：https://join.qq.com/",
                    raw_result={
                        "tool_name": "external.web_search",
                        "ok": True,
                        "result": {"answer": "腾讯校招官网：https://join.qq.com/"},
                    },
                )

        runtime = AgentRuntime(
            registry=AgentCapabilityRegistry(
                [
                    AgentCapabilityDefinition(
                        capability_id="external.web_search",
                        name="网页搜索",
                        description="通过 SDK agent 搜索公开网页。",
                        executor_id="claude-sdk-agent",
                        input_schema={"type": "object", "required": ["query"]},
                        output_schema={"type": "object"},
                        risk_level="low",
                    )
                ]
            ),
            executors={"claude-sdk-agent": FlakySdkAgent()},
            retry_policy=AgentRuntimeRetryPolicy(max_attempts=3, base_delay_seconds=0),
        )

        result = runtime.call(
            AgentTask(
                capability_id="external.web_search",
                goal="查腾讯校招官网",
                input_payload={"query": "腾讯 校园招聘 官网"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="task-1"),
        )

        self.assertEqual("succeeded", result.status)
        self.assertEqual(3, len(calls))
        retry = result.raw_result["runtime_retry"]
        self.assertTrue(retry["recovered"])
        self.assertEqual(3, retry["attempts"])
        self.assertEqual("APITimeoutError", retry["errors"][0]["error_type"])
        self.assertEqual("claude-sdk-agent", calls[0]["context"].namespace)

    def test_runtime_retries_transient_tool_handler_exception_before_succeeding(self) -> None:
        from app.agent_runtime.agent_as_tool import (
            TOOL_REGISTRY_EXECUTOR_ID,
            AgentCapabilityRegistry,
            AgentRuntime,
            AgentRuntimeContext,
            AgentRuntimeRetryPolicy,
            AgentTask,
            ToolRegistryAgentExecutor,
        )
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry

        calls = []

        def flaky_handler(session, **arguments):
            calls.append({"session": session, "arguments": arguments})
            if len(calls) < 3:
                raise TimeoutError("HTTP read timed out")
            return {
                "tool_name": "xiaohongshu-mcp.search_feeds",
                "ok": True,
                "result": {"items": [{"title": "2027 秋招 Java"}]},
            }

        tool_registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="xiaohongshu-mcp.search_feeds",
                    description="搜索小红书公开笔记。",
                    input_schema={"type": "object", "required": ["keyword"]},
                    output_schema={"type": "object"},
                    handler=flaky_handler,
                )
            ]
        )
        runtime = AgentRuntime(
            registry=AgentCapabilityRegistry.from_tool_registry(
                tool_registry,
                default_executor_id=TOOL_REGISTRY_EXECUTOR_ID,
            ),
            executors={
                TOOL_REGISTRY_EXECUTOR_ID: ToolRegistryAgentExecutor(
                    tool_registry,
                    session_provider=lambda context: f"db-session-for-{context.session_id}",
                )
            },
            retry_policy=AgentRuntimeRetryPolicy(max_attempts=3, base_delay_seconds=0),
        )

        result = runtime.call(
            AgentTask(
                capability_id="xiaohongshu-mcp.search_feeds",
                goal="查小红书秋招笔记",
                input_payload={"keyword": "2027 秋招 Java"},
            ),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="task-1"),
        )

        self.assertEqual("succeeded", result.status)
        self.assertEqual(3, len(calls))
        self.assertTrue(result.raw_result["runtime_retry"]["recovered"])
        self.assertEqual("TimeoutError", result.raw_result["runtime_retry"]["errors"][0]["error_type"])

    def test_runtime_does_not_retry_non_transient_business_failure(self) -> None:
        from app.agent_runtime.agent_as_tool import (
            AgentCapabilityDefinition,
            AgentCapabilityRegistry,
            AgentRuntime,
            AgentRuntimeContext,
            AgentRuntimeRetryPolicy,
            AgentTask,
            StandardAgentResult,
        )

        calls = []

        class BusinessFailureAgent:
            def call(self, task: AgentTask, context: AgentRuntimeContext) -> StandardAgentResult:
                calls.append(task)
                return StandardAgentResult(
                    status="failed",
                    summary="缺少简历正文",
                    raw_result={"tool_name": "resume.tailor", "ok": False, "error": "RESUME_TEXT_REQUIRED"},
                )

        runtime = AgentRuntime(
            registry=AgentCapabilityRegistry(
                [
                    AgentCapabilityDefinition(
                        capability_id="resume.tailor",
                        name="简历优化",
                        description="根据目标岗位优化简历。",
                        executor_id="openai-sdk-agent",
                        input_schema={"type": "object", "required": ["resume_text"]},
                        output_schema={"type": "object"},
                    )
                ]
            ),
            executors={"openai-sdk-agent": BusinessFailureAgent()},
            retry_policy=AgentRuntimeRetryPolicy(max_attempts=3, base_delay_seconds=0),
        )

        result = runtime.call(
            AgentTask(capability_id="resume.tailor", goal="优化简历", input_payload={"resume_text": ""}),
            AgentRuntimeContext(session_id="session-1", run_id="run-1", task_id="task-1"),
        )

        self.assertEqual("failed", result.status)
        self.assertEqual(1, len(calls))
        self.assertNotIn("runtime_retry", result.raw_result)


if __name__ == "__main__":
    unittest.main()
