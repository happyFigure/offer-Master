import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class SdkAgentDelegationPolicyTest(unittest.TestCase):
    def test_policy_separates_child_tools_from_runtime_retained_tools(self) -> None:
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY
        from app.agent_runtime.sdk_agents.delegation_policy import build_sdk_agent_delegation_policy
        from app.agent_runtime.tool_registry import (
            ARTIFACT_EXPORT_TOOL,
            EXTERNAL_WEB_SEARCH_TOOL,
            FILESYSTEM_READ_FILE_TOOL,
            FILESYSTEM_WRITE_TEXT_TOOL,
            SKILL_LIST_ACTIONS_TOOL,
            SKILL_LIST_TOOL,
            SKILL_READ_TOOL,
            AgentToolDefinition,
            AgentToolRegistry,
            AgentToolRiskLevel,
            create_skill_lazy_agent_tool_definitions,
        )
        from app.core.config import Settings

        registry = AgentToolRegistry(
            create_skill_lazy_agent_tool_definitions()
            + [
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search public web.",
                    input_schema={"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"ok": True},
                    risk_level=AgentToolRiskLevel.LOW,
                    requires_confirmation=False,
                ),
                AgentToolDefinition(
                    name=FILESYSTEM_READ_FILE_TOOL,
                    description="Read sandbox file.",
                    input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"ok": True},
                    risk_level=AgentToolRiskLevel.LOW,
                    requires_confirmation=False,
                ),
                AgentToolDefinition(
                    name=FILESYSTEM_WRITE_TEXT_TOOL,
                    description="Write sandbox output.",
                    input_schema={"type": "object", "required": ["path", "text"], "properties": {"path": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"ok": True},
                    risk_level=AgentToolRiskLevel.HIGH,
                    requires_confirmation=True,
                ),
                AgentToolDefinition(
                    name=ARTIFACT_EXPORT_TOOL,
                    description="Export approved sandbox artifact.",
                    input_schema={
                        "type": "object",
                        "required": ["artifact_uri", "destination"],
                        "properties": {"artifact_uri": {"type": "string"}, "destination": {"type": "string"}},
                    },
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"ok": True},
                    risk_level=AgentToolRiskLevel.HIGH,
                    requires_confirmation=True,
                ),
            ]
        )

        with patch.dict(
            os.environ,
            {
                "JOBPILOT_SDK_AGENT_ENABLE_WEB_RESEARCH": "true",
                "JOBPILOT_SDK_AGENT_ENABLE_FILE_ANALYSIS": "true",
                "JOBPILOT_SDK_AGENT_ENABLE_MUTATION_TOOLS": "true",
                "JOBPILOT_SDK_AGENT_SANDBOX_MODE": "temp_copy",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        policy = build_sdk_agent_delegation_policy(settings, tool_registry=registry)

        self.assertEqual(
            (EXTERNAL_WEB_SEARCH_TOOL, SKILL_LIST_TOOL, SKILL_LIST_ACTIONS_TOOL, SKILL_READ_TOOL, FILESYSTEM_READ_FILE_TOOL, FILESYSTEM_WRITE_TEXT_TOOL),
            policy.internal_tool_names,
        )
        self.assertEqual(
            (EXTERNAL_WEB_SEARCH_TOOL, FILESYSTEM_SKILL_CAPABILITY),
            policy.exposed_capability_names,
        )
        self.assertEqual((ARTIFACT_EXPORT_TOOL,), policy.runtime_retained_tool_names)
        self.assertNotIn(ARTIFACT_EXPORT_TOOL, policy.internal_tool_names)
        self.assertNotIn(FILESYSTEM_READ_FILE_TOOL, policy.exposed_capability_names)
        self.assertNotIn(FILESYSTEM_WRITE_TEXT_TOOL, policy.exposed_capability_names)
        self.assertEqual("runtime_confirmation_boundary", policy.reasons_by_tool[ARTIFACT_EXPORT_TOOL])
        self.assertEqual("child_agent_skill_orchestration", policy.reasons_by_tool[FILESYSTEM_SKILL_CAPABILITY])
        self.assertEqual("child_internal_filesystem_tool", policy.reasons_by_tool[FILESYSTEM_READ_FILE_TOOL])

    def test_executor_bundle_uses_policy_metadata_for_sdk_runner(self) -> None:
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY, OPENAI_SDK_AGENT_EXECUTOR_ID
        from app.agent_runtime.external_tasks.configured import build_agent_runtime_executor_bundle
        from app.agent_runtime.tool_registry import (
            ARTIFACT_EXPORT_TOOL,
            EXTERNAL_WEB_SEARCH_TOOL,
            FILESYSTEM_READ_FILE_TOOL,
            FILESYSTEM_WRITE_TEXT_TOOL,
            SKILL_LIST_ACTIONS_TOOL,
            SKILL_LIST_TOOL,
            SKILL_READ_TOOL,
            AgentToolDefinition,
            AgentToolRegistry,
            AgentToolRiskLevel,
            create_skill_lazy_agent_tool_definitions,
        )
        from app.core.config import Settings

        registry = AgentToolRegistry(
            create_skill_lazy_agent_tool_definitions()
            + [
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search public web.",
                    input_schema={"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"ok": True},
                    risk_level=AgentToolRiskLevel.LOW,
                ),
                AgentToolDefinition(
                    name=FILESYSTEM_READ_FILE_TOOL,
                    description="Read sandbox file.",
                    input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"ok": True},
                    risk_level=AgentToolRiskLevel.LOW,
                ),
                AgentToolDefinition(
                    name=FILESYSTEM_WRITE_TEXT_TOOL,
                    description="Write sandbox output.",
                    input_schema={"type": "object", "required": ["path", "text"], "properties": {"path": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"ok": True},
                    risk_level=AgentToolRiskLevel.HIGH,
                    requires_confirmation=True,
                ),
                AgentToolDefinition(
                    name=ARTIFACT_EXPORT_TOOL,
                    description="Export approved sandbox artifact.",
                    input_schema={
                        "type": "object",
                        "required": ["artifact_uri", "destination"],
                        "properties": {"artifact_uri": {"type": "string"}, "destination": {"type": "string"}},
                    },
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"ok": True},
                    risk_level=AgentToolRiskLevel.HIGH,
                    requires_confirmation=True,
                ),
            ]
        )

        with patch.dict(
            os.environ,
            {
                "JOBPILOT_EXTERNAL_AGENT_AUTO_DISPATCH": "true",
                "JOBPILOT_EXTERNAL_WEB_SEARCH_PROVIDER": "bailian",
                "JOBPILOT_OPENAI_SDK_AGENT_ENABLED": "true",
                "JOBPILOT_OPENAI_SDK_AGENT_MODE": "agents_sdk",
                "JOBPILOT_OPENAI_SDK_AGENT_API_KEY": "sk-openai-test",
                "JOBPILOT_OPENAI_SDK_AGENT_MODEL": "gpt-test",
                "JOBPILOT_SDK_AGENT_ENABLE_WEB_RESEARCH": "true",
                "JOBPILOT_SDK_AGENT_ENABLE_FILE_ANALYSIS": "true",
                "JOBPILOT_SDK_AGENT_ENABLE_MUTATION_TOOLS": "true",
                "JOBPILOT_SDK_AGENT_SANDBOX_MODE": "temp_copy",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        executors, capability_executor_ids = build_agent_runtime_executor_bundle(settings, tool_registry=registry)
        adapter = executors[OPENAI_SDK_AGENT_EXECUTOR_ID]._adapter

        self.assertEqual(
            [EXTERNAL_WEB_SEARCH_TOOL, SKILL_LIST_TOOL, SKILL_LIST_ACTIONS_TOOL, SKILL_READ_TOOL, FILESYSTEM_READ_FILE_TOOL, FILESYSTEM_WRITE_TEXT_TOOL],
            adapter.allowed_tools,
        )
        self.assertEqual(OPENAI_SDK_AGENT_EXECUTOR_ID, capability_executor_ids[EXTERNAL_WEB_SEARCH_TOOL])
        self.assertEqual(OPENAI_SDK_AGENT_EXECUTOR_ID, capability_executor_ids[FILESYSTEM_SKILL_CAPABILITY])
        self.assertNotIn(FILESYSTEM_READ_FILE_TOOL, capability_executor_ids)
        self.assertNotIn(FILESYSTEM_WRITE_TEXT_TOOL, capability_executor_ids)
        self.assertNotIn(ARTIFACT_EXPORT_TOOL, capability_executor_ids)
        self.assertEqual([ARTIFACT_EXPORT_TOOL], adapter.risk_policy["delegation"]["runtime_retained_tools"])

    def test_policy_exposes_only_high_level_chrome_and_readonly_dbx_agents(self) -> None:
        from app.agent_runtime.sdk_agents.delegation_policy import build_sdk_agent_delegation_policy
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel
        from app.core.config import Settings

        chrome_tool = "mcp.chrome.list_pages"
        dbx_tool = "mcp.dbx.dbx_list_connections"
        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name=chrome_tool,
                    description="List pages from Google Chrome MCP.",
                    input_schema={"type": "object", "additionalProperties": True},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"tool_name": chrome_tool, "ok": True},
                    risk_level=AgentToolRiskLevel.LOW,
                    requires_confirmation=False,
                ),
                AgentToolDefinition(
                    name=dbx_tool,
                    description="List DBX connections from dbx MCP.",
                    input_schema={"type": "object", "additionalProperties": True},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"tool_name": dbx_tool, "ok": True},
                    risk_level=AgentToolRiskLevel.LOW,
                    requires_confirmation=False,
                ),
            ]
        )

        with patch.dict(
            os.environ,
            {
                "JOBPILOT_SDK_AGENT_ENABLE_CHROME_MCP": "true",
                "JOBPILOT_SDK_AGENT_ENABLE_DBX_MCP": "true",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        policy = build_sdk_agent_delegation_policy(settings, tool_registry=registry)

        self.assertIn(chrome_tool, policy.internal_tool_names)
        self.assertIn(dbx_tool, policy.internal_tool_names)
        self.assertEqual(
            ("agent.google_chrome", "agent.dbx_readonly"),
            policy.exposed_capability_names,
        )
        self.assertNotIn(chrome_tool, policy.exposed_capability_names)
        self.assertNotIn(dbx_tool, policy.exposed_capability_names)
        self.assertEqual((chrome_tool,), policy.tools_by_capability["agent.google_chrome"])
        self.assertEqual((dbx_tool,), policy.tools_by_capability["agent.dbx_readonly"])
        self.assertEqual("child_agent_mcp_tool", policy.reasons_by_tool[chrome_tool])
        self.assertEqual("child_agent_mcp_tool", policy.reasons_by_tool[dbx_tool])

    def test_chrome_and_dbx_agents_survive_runtime_catalog_and_are_offered_on_normal_chat(self) -> None:
        from types import SimpleNamespace

        from app.agent_runtime.agent_as_tool import AgentCapabilityDefinition, OPENAI_SDK_AGENT_EXECUTOR_ID
        from app.agent_runtime.context.capability_catalog import CapabilityCatalog
        from app.agent_runtime.context.context_pack import ContextPackBuilder
        from app.agent_runtime.graph_factory import (
            _build_native_tool_schema_bundle,
            _runtime_capability_registry,
        )
        from app.agent_runtime.understanding.intent_detector import HybridIntentDetector
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel

        class DeclaredAgent:
            executor_id = OPENAI_SDK_AGENT_EXECUTOR_ID

            def capabilities(self):
                return [
                    AgentCapabilityDefinition(
                        capability_id="agent.google_chrome",
                        name="Google Chrome Agent",
                        description="Delegate browser tasks to Chrome MCP.",
                        executor_id=self.executor_id,
                        input_schema={"type": "object", "required": ["task"], "properties": {"task": {"type": "string"}}},
                        output_schema={"type": "object"},
                        kind="agent",
                        always_available=True,
                        allowed_source_types=frozenset({"agent_chat"}),
                    ),
                    AgentCapabilityDefinition(
                        capability_id="agent.dbx_readonly",
                        name="DBX Read-only Agent",
                        description="Delegate read-only database tasks to DBX MCP.",
                        executor_id=self.executor_id,
                        input_schema={"type": "object", "required": ["task"], "properties": {"task": {"type": "string"}}},
                        output_schema={"type": "object"},
                        kind="agent",
                        always_available=True,
                        allowed_source_types=frozenset({"agent_chat"}),
                    ),
                ]

        internal_tools = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="mcp.chrome.list_pages",
                    description="Internal Chrome MCP tool.",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {},
                    risk_level=AgentToolRiskLevel.LOW,
                ),
                AgentToolDefinition(
                    name="mcp.dbx.dbx_list_connections",
                    description="Internal DBX MCP tool.",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {},
                    risk_level=AgentToolRiskLevel.LOW,
                ),
            ]
        )
        dependencies = SimpleNamespace(
            registry=internal_tools,
            db_session=object(),
            llm_client=object(),
            capability_executor_ids={
                "agent.google_chrome": OPENAI_SDK_AGENT_EXECUTOR_ID,
                "agent.dbx_readonly": OPENAI_SDK_AGENT_EXECUTOR_ID,
            },
            agent_executors={OPENAI_SDK_AGENT_EXECUTOR_ID: DeclaredAgent()},
        )

        capability_registry = _runtime_capability_registry(dependencies)
        catalog = CapabilityCatalog.from_agent_registry(capability_registry)
        context_pack = ContextPackBuilder(catalog).build(HybridIntentDetector(llm_client=None).detect("你好"))
        model_tools = _build_native_tool_schema_bundle(capability_registry, context_pack.allowed_capabilities)
        offered_to_model = set(model_tools["alias_to_tool_name"].values())

        self.assertEqual("agent", capability_registry.get("agent.google_chrome").kind)
        self.assertTrue(capability_registry.get("agent.google_chrome").always_available)
        self.assertIsNone(capability_registry.get("mcp.chrome.list_pages"))
        self.assertIsNone(capability_registry.get("mcp.dbx.dbx_list_connections"))
        self.assertIn("agent.google_chrome", context_pack.allowed_capabilities)
        self.assertIn("agent.dbx_readonly", context_pack.allowed_capabilities)
        self.assertIn("agent.google_chrome", offered_to_model)
        self.assertIn("agent.dbx_readonly", offered_to_model)

    def test_runtime_passes_event_sink_and_selected_capability_into_child_agent(self) -> None:
        from types import SimpleNamespace

        from app.agent_runtime.agent_as_tool import AgentCapabilityDefinition, OPENAI_SDK_AGENT_EXECUTOR_ID, StandardAgentResult
        from app.agent_runtime.graph_factory import AgentRunCommand, _run_agent_tool_through_runtime
        from app.agent_runtime.state import AgentState
        from app.agent_runtime.tool_registry import AgentToolRegistry

        observed = {}
        events = []

        class CapturingAgent:
            executor_id = OPENAI_SDK_AGENT_EXECUTOR_ID

            def capabilities(self):
                return [
                    AgentCapabilityDefinition(
                        capability_id="agent.google_chrome",
                        name="Google Chrome Agent",
                        description="Delegate browser tasks to Chrome MCP.",
                        executor_id=self.executor_id,
                        input_schema={"type": "object", "required": ["task"], "properties": {"task": {"type": "string"}}},
                        output_schema={"type": "object"},
                        kind="agent",
                        always_available=True,
                        allowed_source_types=frozenset({"agent_chat"}),
                    )
                ]

            def call(self, task, context):
                observed["task"] = task
                observed["context"] = context
                context.event_sink({"event_type": "child_event"})
                return StandardAgentResult(status="succeeded", summary="done")

        class LocalToolAgent:
            executor_id = "local-tool-executor"

            def capabilities(self):
                return [
                    AgentCapabilityDefinition(
                        capability_id="local.company_database_overview",
                        name="本地企业库概览",
                        description="Read the local company database overview.",
                        executor_id=self.executor_id,
                        input_schema={"type": "object", "properties": {}},
                        output_schema={"type": "object"},
                        kind="tool",
                        allowed_source_types=frozenset({"agent_chat"}),
                    )
                ]

            def call(self, _task, _context):
                return StandardAgentResult(status="succeeded", summary="local result")

        dependencies = SimpleNamespace(
            registry=AgentToolRegistry(),
            capability_executor_ids={
                "agent.google_chrome": OPENAI_SDK_AGENT_EXECUTOR_ID,
                "local.company_database_overview": "local-tool-executor",
            },
            agent_executors={
                OPENAI_SDK_AGENT_EXECUTOR_ID: CapturingAgent(),
                "local-tool-executor": LocalToolAgent(),
            },
            db_session=None,
            event_sink=events.append,
        )
        state = AgentState(
            session_id="s1",
            workflow_run_id="w1",
            agent_run_id="a1",
            user_message="Inspect the current browser page.",
            current_step="maybe_tool",
        )

        result = _run_agent_tool_through_runtime(
            AgentRunCommand(session_id="s1", user_message=state.user_message, requested_tool_name="agent.google_chrome"),
            state=state,
            dependencies=dependencies,
            tool_input={"task": "Inspect the current browser page."},
        )

        self.assertEqual("succeeded", result.status)
        self.assertEqual("agent.google_chrome", observed["context"].capability_id)
        self.assertIsNotNone(observed["context"].event_sink)
        self.assertEqual(
            ["subagent_started", "child_event", "subagent_finished"],
            [event["event_type"] for event in events],
        )
        self.assertEqual("agent.google_chrome", events[0]["capability"])
        self.assertTrue(events[0]["delegation_id"].startswith("delegation:w1:"))
        self.assertEqual("running", events[0]["status"])
        self.assertEqual("succeeded", events[-1]["status"])

        events.clear()
        local_result = _run_agent_tool_through_runtime(
            AgentRunCommand(
                session_id="s1",
                user_message=state.user_message,
                requested_tool_name="local.company_database_overview",
            ),
            state=state,
            dependencies=dependencies,
            tool_input={},
        )

        self.assertEqual("succeeded", local_result.status)
        self.assertEqual([], events)


if __name__ == "__main__":
    unittest.main()
