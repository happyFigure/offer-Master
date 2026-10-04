import asyncio
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class SdkAgentToolGatewayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="sdk-agent-tool-gateway-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_gateway_wraps_only_low_risk_readonly_tools(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext
        from app.agent_runtime.sdk_agents.tool_gateway import SdkAgentToolGateway
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel

        class FakeFunctionTool:
            def __init__(self, **payload):
                self.payload = payload

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="filesystem.read_file",
                    description="Read a file without modifying it.",
                    input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"tool_name": "filesystem.read_file", "ok": True, "result": {}},
                    risk_level=AgentToolRiskLevel.LOW,
                    requires_confirmation=False,
                    allowed_source_types=frozenset({"agent_chat"}),
                ),
                AgentToolDefinition(
                    name="filesystem.delete_path",
                    description="Delete a file.",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"tool_name": "filesystem.delete_path", "ok": True},
                    risk_level=AgentToolRiskLevel.HIGH,
                    requires_confirmation=True,
                    allowed_source_types=frozenset({"agent_chat"}),
                ),
            ]
        )

        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(FunctionTool=FakeFunctionTool)}):
            tools = SdkAgentToolGateway(registry).build_tools(
                ["filesystem.read_file", "filesystem.delete_path"],
                AgentRuntimeContext(session_id="s1", run_id="r1", task_id="t1", permission_scope={"source_type": "agent_chat"}),
            )

        self.assertEqual(1, len(tools))
        self.assertEqual("filesystem_read_file", tools[0].payload["name"])
        self.assertIn("Registry tool name: filesystem.read_file", tools[0].payload["description"])
        self.assertEqual({"filesystem_read_file": "filesystem.read_file"}, SdkAgentToolGateway(registry).safe_aliases(["filesystem.read_file"]))

    def test_gateway_invokes_existing_handler_with_session_and_compresses_output(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext
        from app.agent_runtime.sdk_agents.tool_gateway import SdkAgentToolGateway
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel

        calls = []

        class FakeFunctionTool:
            def __init__(self, **payload):
                self.payload = payload
                self.on_invoke_tool = payload["on_invoke_tool"]

        def read_handler(session, **arguments):
            calls.append({"session": session, "arguments": arguments})
            return {
                "tool_name": "filesystem.read_file",
                "ok": True,
                "result": {"content": "A" * 5000, "path": arguments["path"]},
            }

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="filesystem.read_file",
                    description="Read a file without modifying it.",
                    input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=read_handler,
                    risk_level=AgentToolRiskLevel.LOW,
                    requires_confirmation=False,
                )
            ]
        )

        context = AgentRuntimeContext(session_id="s1", run_id="r1", task_id="t1")
        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(FunctionTool=FakeFunctionTool)}):
            tool = SdkAgentToolGateway(
                registry,
                session_provider=lambda _context: "db-session",
                max_observation_chars=200,
            ).build_tools(["filesystem.read_file"], context)[0]

        output = asyncio.run(tool.on_invoke_tool(None, json.dumps({"path": "resume.md"})))
        payload = json.loads(output)

        self.assertEqual([{"session": "db-session", "arguments": {"path": "resume.md"}}], calls)
        self.assertEqual("filesystem.read_file", payload["tool_name"])
        self.assertTrue(payload["ok"])
        self.assertLessEqual(len(payload["observation"]), 260)
        self.assertNotIn("A" * 1000, output)
        self.assertTrue(payload["truncated"])

    def test_gateway_returns_tool_error_for_invalid_json_arguments(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext
        from app.agent_runtime.sdk_agents.tool_gateway import SdkAgentToolGateway
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry

        class FakeFunctionTool:
            def __init__(self, **payload):
                self.on_invoke_tool = payload["on_invoke_tool"]

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="external.web_search",
                    description="Search the web.",
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"tool_name": "external.web_search", "ok": True},
                )
            ]
        )

        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(FunctionTool=FakeFunctionTool)}):
            tool = SdkAgentToolGateway(registry).build_tools(
                ["external.web_search"],
                AgentRuntimeContext(session_id="s1", run_id="r1", task_id="t1"),
            )[0]

        payload = json.loads(asyncio.run(tool.on_invoke_tool(None, "not json")))

        self.assertFalse(payload["ok"])
        self.assertEqual("INVALID_TOOL_ARGUMENTS", payload["error"])
        self.assertEqual("external.web_search", payload["tool_name"])

    def test_gateway_maps_sandbox_read_path_before_handler_invocation(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext
        from app.agent_runtime.sdk_agents.sandbox import SdkAgentSandboxManager
        from app.agent_runtime.sdk_agents.tool_gateway import SdkAgentToolGateway
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel

        calls = []

        class FakeFunctionTool:
            def __init__(self, **payload):
                self.on_invoke_tool = payload["on_invoke_tool"]

        manager = SdkAgentSandboxManager(base_dir=self.tmpdir / "runs")
        workspace = manager.prepare_workspace(run_id="run_001")

        def read_handler(_session, **arguments):
            calls.append(arguments)
            return {"tool_name": "filesystem.read_file", "ok": True, "result": {"path": arguments["path"]}}

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="filesystem.read_file",
                    description="Read sandbox file.",
                    input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=read_handler,
                    risk_level=AgentToolRiskLevel.LOW,
                    requires_confirmation=False,
                )
            ]
        )
        context = AgentRuntimeContext(
            session_id="s1",
            run_id="r1",
            task_id="t1",
            metadata={"sdk_agent_sandbox": workspace.to_metadata()},
        )

        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(FunctionTool=FakeFunctionTool)}):
            tool = SdkAgentToolGateway(registry).build_tools(["filesystem.read_file"], context)[0]

        payload = json.loads(asyncio.run(tool.on_invoke_tool(None, json.dumps({"path": "input/resume.md"}))))

        self.assertTrue(payload["ok"])
        self.assertEqual([{"path": str(workspace.input_dir / "resume.md")}], calls)

    def test_gateway_allows_sandbox_output_write_for_mutation_tool(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext
        from app.agent_runtime.sdk_agents.sandbox import SdkAgentSandboxManager
        from app.agent_runtime.sdk_agents.tool_gateway import SdkAgentToolGateway
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel

        calls = []

        class FakeFunctionTool:
            def __init__(self, **payload):
                self.on_invoke_tool = payload["on_invoke_tool"]

        manager = SdkAgentSandboxManager(base_dir=self.tmpdir / "runs")
        workspace = manager.prepare_workspace(run_id="run_001")

        def write_handler(_session, **arguments):
            calls.append(arguments)
            return {"tool_name": "filesystem.write_text", "ok": True, "result": {"path": arguments["path"]}}

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="filesystem.write_text",
                    description="Write sandbox output.",
                    input_schema={"type": "object", "required": ["path", "text"], "properties": {"path": {"type": "string"}, "text": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=write_handler,
                    risk_level=AgentToolRiskLevel.HIGH,
                    requires_confirmation=True,
                )
            ]
        )
        context = AgentRuntimeContext(
            session_id="s1",
            run_id="r1",
            task_id="t1",
            metadata={"sdk_agent_sandbox": workspace.to_metadata()},
        )

        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(FunctionTool=FakeFunctionTool)}):
            tool = SdkAgentToolGateway(registry).build_tools(["filesystem.write_text"], context)[0]

        payload = json.loads(asyncio.run(tool.on_invoke_tool(None, json.dumps({"path": "output/resume.md", "text": "final"}))))

        self.assertTrue(payload["ok"])
        self.assertEqual([{"path": str(workspace.output_dir / "resume.md"), "text": "final"}], calls)

    def test_gateway_rejects_sandbox_path_escape_without_calling_handler(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext
        from app.agent_runtime.sdk_agents.sandbox import SdkAgentSandboxManager
        from app.agent_runtime.sdk_agents.tool_gateway import SdkAgentToolGateway
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel

        calls = []

        class FakeFunctionTool:
            def __init__(self, **payload):
                self.on_invoke_tool = payload["on_invoke_tool"]

        manager = SdkAgentSandboxManager(base_dir=self.tmpdir / "runs")
        workspace = manager.prepare_workspace(run_id="run_001")
        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="filesystem.delete_path",
                    description="Delete sandbox output.",
                    input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=lambda _session, **arguments: calls.append(arguments) or {"tool_name": "filesystem.delete_path", "ok": True},
                    risk_level=AgentToolRiskLevel.HIGH,
                    requires_confirmation=True,
                )
            ]
        )
        context = AgentRuntimeContext(
            session_id="s1",
            run_id="r1",
            task_id="t1",
            metadata={"sdk_agent_sandbox": workspace.to_metadata()},
        )

        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(FunctionTool=FakeFunctionTool)}):
            tool = SdkAgentToolGateway(registry).build_tools(["filesystem.delete_path"], context)[0]

        payload = json.loads(asyncio.run(tool.on_invoke_tool(None, json.dumps({"path": r"C:\Users\phoenix\Documents\resume.docx"}))))

        self.assertFalse(payload["ok"])
        self.assertEqual("PATH_OUTSIDE_SANDBOX", payload["error"])
        self.assertEqual([], calls)

    def test_gateway_rejects_sandbox_input_mutation_without_calling_handler(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext
        from app.agent_runtime.sdk_agents.sandbox import SdkAgentSandboxManager
        from app.agent_runtime.sdk_agents.tool_gateway import SdkAgentToolGateway
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel

        calls = []

        class FakeFunctionTool:
            def __init__(self, **payload):
                self.on_invoke_tool = payload["on_invoke_tool"]

        manager = SdkAgentSandboxManager(base_dir=self.tmpdir / "runs")
        workspace = manager.prepare_workspace(run_id="run_001")
        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name="filesystem.replace_text",
                    description="Replace sandbox text.",
                    input_schema={"type": "object", "required": ["path", "old_text", "new_text"], "properties": {"path": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=lambda _session, **arguments: calls.append(arguments) or {"tool_name": "filesystem.replace_text", "ok": True},
                    risk_level=AgentToolRiskLevel.HIGH,
                    requires_confirmation=True,
                )
            ]
        )
        context = AgentRuntimeContext(
            session_id="s1",
            run_id="r1",
            task_id="t1",
            metadata={"sdk_agent_sandbox": workspace.to_metadata()},
        )

        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(FunctionTool=FakeFunctionTool)}):
            tool = SdkAgentToolGateway(registry).build_tools(["filesystem.replace_text"], context)[0]

        payload = json.loads(
            asyncio.run(
                tool.on_invoke_tool(
                    None,
                    json.dumps({"path": "input/resume.md", "old_text": "A", "new_text": "B"}),
                )
            )
        )

        self.assertFalse(payload["ok"])
        self.assertEqual("PATH_OUTSIDE_SANDBOX", payload["error"])
        self.assertEqual([], calls)

    def test_gateway_exposes_progressive_skill_read_tools_to_child_agent(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext
        from app.agent_runtime.sdk_agents.tool_gateway import SdkAgentToolGateway
        from app.agent_runtime.tool_registry import SKILL_LIST_TOOL, create_skill_lazy_agent_tool_definitions, AgentToolRegistry

        class FakeFunctionTool:
            def __init__(self, **payload):
                self.payload = payload
                self.on_invoke_tool = payload["on_invoke_tool"]

        registry = AgentToolRegistry(create_skill_lazy_agent_tool_definitions())
        context = AgentRuntimeContext(session_id="s1", run_id="r1", task_id="t1", permission_scope={"source_type": "agent_chat"})

        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(FunctionTool=FakeFunctionTool)}):
            tool = SdkAgentToolGateway(registry).build_tools([SKILL_LIST_TOOL], context)[0]

        payload = json.loads(asyncio.run(tool.on_invoke_tool(None, json.dumps({"query": "filesystem"}))))

        self.assertEqual("skill_list", payload["tool_name"])
        self.assertTrue(payload["ok"])
        self.assertIn("observation", payload)

    def test_gateway_exposes_registered_mcp_tools_to_child_agent(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext
        from app.agent_runtime.sdk_agents.tool_gateway import SdkAgentToolGateway
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel

        class FakeFunctionTool:
            def __init__(self, **payload):
                self.payload = payload
                self.on_invoke_tool = payload["on_invoke_tool"]

        calls = []
        chrome_tool = "mcp.chrome.list_pages"

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name=chrome_tool,
                    description="List pages from Google Chrome MCP.",
                    input_schema={"type": "object", "additionalProperties": True},
                    output_schema={"type": "object"},
                    handler=lambda _session, **arguments: calls.append(arguments) or {"tool_name": chrome_tool, "ok": True, "result": {"pages": []}},
                    risk_level=AgentToolRiskLevel.LOW,
                    requires_confirmation=False,
                )
            ]
        )
        context = AgentRuntimeContext(session_id="s1", run_id="r1", task_id="t1", permission_scope={"source_type": "agent_chat"})

        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(FunctionTool=FakeFunctionTool)}):
            tool = SdkAgentToolGateway(registry).build_tools([chrome_tool], context)[0]

        payload = json.loads(asyncio.run(tool.on_invoke_tool(None, json.dumps({"pageId": 0}))))

        self.assertEqual("mcp.chrome.list_pages", payload["tool_name"])
        self.assertTrue(payload["ok"])
        self.assertEqual([{"pageId": 0}], calls)

    def test_gateway_emits_nested_mcp_events_and_marks_confirmation_tools(self) -> None:
        from types import SimpleNamespace
        from app.agent_runtime.sdk_agents.tool_gateway import SdkAgentToolGateway
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel

        class FakeFunctionTool:
            def __init__(self, **payload):
                self.payload = payload
                self.on_invoke_tool = payload["on_invoke_tool"]

        events = []
        calls = []
        tool_name = "mcp.chrome.click"
        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name=tool_name,
                    description="Click a browser element.",
                    input_schema={"type": "object", "properties": {"selector": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=lambda _session, **arguments: calls.append(arguments) or {"ok": True, "result": {}},
                    risk_level=AgentToolRiskLevel.HIGH,
                    requires_confirmation=True,
                )
            ]
        )
        context = SimpleNamespace(
            session_id="s1",
            run_id="r1",
            task_id="t1",
            capability_id="agent.google_chrome",
            child_agent_name="GoogleChromeAgent",
            executor_id="test-specialist-agent",
            event_sink=events.append,
            permission_scope={"source_type": "agent_chat"},
            metadata={"agent_run_id": "a1", "delegation_id": "delegation:w1:1"},
        )

        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(FunctionTool=FakeFunctionTool)}):
            tool = SdkAgentToolGateway(registry).build_tools([tool_name], context)[0]

        self.assertTrue(tool.payload["needs_approval"])
        payload = json.loads(asyncio.run(tool.on_invoke_tool(None, json.dumps({"selector": "#next"}))))
        self.assertTrue(payload["ok"])
        self.assertEqual([{"selector": "#next"}], calls)
        self.assertEqual(["subagent_tool_started", "subagent_tool_finished"], [event["event_type"] for event in events])
        self.assertTrue(all(event["delegation_id"] == "delegation:w1:1" for event in events))
        self.assertTrue(all(event["parent_capability"] == "agent.google_chrome" for event in events))
        self.assertTrue(all(event["parent_agent_name"] == "GoogleChromeAgent" for event in events))
        self.assertTrue(all(event["executor_id"] == "test-specialist-agent" for event in events))

    def test_gateway_invokes_chrome_and_dbx_tools_created_by_mcp_registry(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext
        from app.agent_runtime.sdk_agents.tool_gateway import SdkAgentToolGateway
        from app.agent_runtime.tool_registry import AgentToolRegistry, create_mcp_agent_tool_definitions
        from app.mcp_gateway.client import MCPToolCallResult

        class FakeFunctionTool:
            def __init__(self, **payload):
                self.payload = payload
                self.on_invoke_tool = payload["on_invoke_tool"]

        calls = []

        class FakeMCPClient:
            def call_tool(self, *, tool_name, arguments):
                calls.append({"tool_name": tool_name, "arguments": arguments})
                return MCPToolCallResult(tool_name=tool_name, ok=True, result={"items": []})

        registry = AgentToolRegistry(
            create_mcp_agent_tool_definitions(
                FakeMCPClient(),
                allowed_tool_names=["chrome.list_pages", "dbx.dbx_list_connections"],
            )
        )
        context = AgentRuntimeContext(session_id="s1", run_id="r1", task_id="t1", permission_scope={"source_type": "agent_chat"})

        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(FunctionTool=FakeFunctionTool)}):
            tools = SdkAgentToolGateway(registry).build_tools(["mcp.chrome.list_pages", "mcp.dbx.dbx_list_connections"], context)

        self.assertEqual(["mcp_chrome_list_pages", "mcp_dbx_dbx_list_connections"], [tool.payload["name"] for tool in tools])

        chrome_payload = json.loads(asyncio.run(tools[0].on_invoke_tool(None, json.dumps({"active": True}))))
        dbx_payload = json.loads(asyncio.run(tools[1].on_invoke_tool(None, json.dumps({}))))

        self.assertTrue(chrome_payload["ok"])
        self.assertTrue(dbx_payload["ok"])
        self.assertEqual(
            [
                {"tool_name": "chrome.list_pages", "arguments": {"active": True}},
                {"tool_name": "dbx.dbx_list_connections", "arguments": {}},
            ],
            calls,
        )

    def test_gateway_exposes_all_registered_dbx_readonly_tools_to_child_agent(self) -> None:
        """The DBX child agent must receive metadata and query tools, not only discovery tools."""
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext
        from app.agent_runtime.sdk_agents.tool_gateway import SdkAgentToolGateway
        from app.agent_runtime.tool_registry import AgentToolRegistry, create_mcp_agent_tool_definitions

        class FakeFunctionTool:
            def __init__(self, **payload):
                self.payload = payload

        class FakeMCPClient:
            def call_tool(self, *, tool_name, arguments):
                return {"tool_name": tool_name, "ok": True, "result": {"arguments": arguments}}

        dbx_tools = [
            "dbx.dbx_list_connections",
            "dbx.dbx_list_databases",
            "dbx.dbx_list_tables",
            "dbx.dbx_get_schema_context",
            "dbx.dbx_describe_table",
            "dbx.dbx_execute_query",
        ]
        registry = AgentToolRegistry(
            create_mcp_agent_tool_definitions(FakeMCPClient(), allowed_tool_names=dbx_tools)
        )
        context = AgentRuntimeContext(
            session_id="s1",
            run_id="r1",
            task_id="t1",
            permission_scope={"source_type": "agent_chat"},
        )

        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(FunctionTool=FakeFunctionTool)}):
            tools = SdkAgentToolGateway(registry).build_tools(
                [f"mcp.{tool_name}" for tool_name in dbx_tools],
                context,
            )

        self.assertEqual(
            [f"mcp_{tool_name.replace('.', '_')}" for tool_name in dbx_tools],
            [tool.payload["name"] for tool in tools],
        )

    def test_gateway_records_runtime_mcp_evidence_without_result_contents(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext
        from app.agent_runtime.sdk_agents.tool_gateway import SdkAgentToolGateway
        from app.agent_runtime.tool_registry import AgentToolRegistry, create_mcp_agent_tool_definitions
        from app.mcp_gateway.client import MCPToolCallResult

        class FakeFunctionTool:
            def __init__(self, **payload):
                self.payload = payload
                self.on_invoke_tool = payload["on_invoke_tool"]

        secret_result = "private database row must not enter runtime evidence"

        class FakeMCPClient:
            def call_tool(self, *, tool_name, arguments):
                return MCPToolCallResult(tool_name=tool_name, ok=True, result={"text": secret_result})

        registry = AgentToolRegistry(
            create_mcp_agent_tool_definitions(
                FakeMCPClient(), allowed_tool_names=["dbx.dbx_list_connections"]
            )
        )
        context = AgentRuntimeContext(
            session_id="s1",
            run_id="r1",
            task_id="t1",
            capability_id="agent.dbx_readonly",
            agent_name="DbxReadOnlyAgent",
            permission_scope={"source_type": "agent_chat"},
        )

        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(FunctionTool=FakeFunctionTool)}):
            tool = SdkAgentToolGateway(registry).build_tools(
                ["mcp.dbx.dbx_list_connections"], context
            )[0]
        payload = json.loads(asyncio.run(tool.on_invoke_tool(None, "{}")))

        self.assertTrue(payload["ok"])
        evidence = context.metadata["_sdk_runtime_mcp_calls"]
        self.assertEqual(1, len(evidence))
        self.assertEqual("mcp.dbx.dbx_list_connections", evidence[0]["tool_name"])
        self.assertEqual("succeeded", evidence[0]["status"])
        self.assertTrue(evidence[0]["call_id"].startswith("sdk-tool:"))
        self.assertNotIn(secret_result, str(evidence))

    def test_dbx_requires_connection_discovery_before_other_calls(self) -> None:
        from app.agent_runtime.agent_as_tool import AgentRuntimeContext
        from app.agent_runtime.sdk_agents.tool_gateway import SdkAgentToolGateway
        from app.agent_runtime.tool_registry import AgentToolRegistry, create_mcp_agent_tool_definitions
        from app.mcp_gateway.client import MCPToolCallResult

        class FakeFunctionTool:
            def __init__(self, **payload):
                self.payload = payload
                self.on_invoke_tool = payload["on_invoke_tool"]

        calls = []

        class FakeMCPClient:
            def call_tool(self, *, tool_name, arguments):
                calls.append(tool_name)
                return MCPToolCallResult(tool_name=tool_name, ok=True, result={"rows": []})

        names = ["dbx.dbx_list_connections", "dbx.dbx_execute_query"]
        registry = AgentToolRegistry(
            create_mcp_agent_tool_definitions(FakeMCPClient(), allowed_tool_names=names)
        )
        context = AgentRuntimeContext(
            session_id="s1",
            run_id="r1",
            task_id="t1",
            capability_id="agent.dbx_readonly",
            agent_name="DbxReadOnlyAgent",
            permission_scope={"source_type": "agent_chat"},
        )

        with patch.dict(sys.modules, {"agents": types.SimpleNamespace(FunctionTool=FakeFunctionTool)}):
            tools = SdkAgentToolGateway(registry).build_tools(
                ["mcp.dbx.dbx_list_connections", "mcp.dbx.dbx_execute_query"], context
            )
        by_name = {tool.payload["description"].split("Registry tool name: ")[-1]: tool for tool in tools}
        early_query = json.loads(
            asyncio.run(by_name["mcp.dbx.dbx_execute_query"].on_invoke_tool(None, json.dumps({"sql": "SELECT 1"})))
        )
        connections = json.loads(
            asyncio.run(by_name["mcp.dbx.dbx_list_connections"].on_invoke_tool(None, "{}"))
        )

        self.assertFalse(early_query["ok"])
        self.assertEqual("DBX_CONNECTION_DISCOVERY_REQUIRED", early_query["error"])
        self.assertTrue(connections["ok"])
        self.assertEqual(["dbx.dbx_list_connections"], calls)


if __name__ == "__main__":
    unittest.main()
