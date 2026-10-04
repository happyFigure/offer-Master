import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class MCPRegistryTests(unittest.TestCase):
    def test_loads_declarative_servers_and_resolves_env_without_logging_secret(self) -> None:
        from app.mcp_gateway.registry_config import load_mcp_server_configs

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mcp.json"
            path.write_text(
                json.dumps(
                    {
                        "servers": [
                            {
                                "id": "dbx",
                                "tool_prefix": "dbx",
                                "command": "node",
                                "args": ["server.js"],
                                "env": {"DBX_TOKEN": "${DBX_TOKEN}"},
                                "enabled_env": "ENABLE_DBX",
                                "allow_tools": ["dbx_list_connections"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            configs = load_mcp_server_configs(
                path,
                environ={"DBX_TOKEN": "secret-value", "ENABLE_DBX": "true"},
            )

        self.assertEqual(("dbx_list_connections",), configs[0].allow_tools)
        self.assertEqual("secret-value", configs[0].env["DBX_TOKEN"])
        self.assertEqual("dbx", configs[0].tool_prefix)

    def test_registry_filters_tools_and_preserves_discovered_schema(self) -> None:
        from app.mcp_gateway.registry import MCPRegistryClient
        from app.mcp_gateway.registry_config import MCPServerConfig

        class FakeTransport:
            async def list_tools(self, server):
                return [
                    {
                        "name": "dbx_list_connections",
                        "description": "List connections",
                        "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer"}}},
                    },
                    {"name": "dbx_commit", "description": "Write", "inputSchema": {"type": "object"}},
                ]

            async def call_tool(self, server, *, tool_name, arguments):
                from app.mcp_gateway.client import MCPToolCallResult

                return MCPToolCallResult(tool_name=tool_name, ok=True, result={"tool": tool_name})

        client = MCPRegistryClient(
            servers=(
                MCPServerConfig(
                    id="dbx",
                    tool_prefix="dbx",
                    command="node",
                    allow_tools=("dbx_list_connections",),
                ),
            ),
            transport=FakeTransport(),
        )

        discovered = client.discover_tools()

        self.assertEqual(["dbx.dbx_list_connections"], [tool["name"] for tool in discovered])
        self.assertEqual(
            {"type": "object", "properties": {"limit": {"type": "integer"}}},
            client.tool_metadata("dbx.dbx_list_connections")["inputSchema"],
        )
        self.assertEqual("ready", client.statuses()[0]["status"])

    def test_registry_rejects_tool_outside_declared_allowlist(self) -> None:
        from app.mcp_gateway.registry import MCPRegistryClient
        from app.mcp_gateway.registry_config import MCPServerConfig

        class FakeTransport:
            async def list_tools(self, server):
                return []

            async def call_tool(self, server, *, tool_name, arguments):
                raise AssertionError("denied tool must never reach transport")

        client = MCPRegistryClient(
            servers=(MCPServerConfig(id="chrome", tool_prefix="chrome", command="node", allow_tools=("list_pages",)),),
            transport=FakeTransport(),
        )

        result = client.call_tool(tool_name="chrome.navigate_page", arguments={})

        self.assertFalse(result.ok)
        self.assertEqual("MCP_TOOL_POLICY_DENIED", result.error)

    def test_failed_discovery_keeps_declared_tools_but_blocks_execution(self) -> None:
        from app.mcp_gateway.registry import MCPRegistryClient
        from app.mcp_gateway.registry_config import MCPServerConfig

        class FakeTransport:
            async def list_tools(self, server):
                return []

            async def call_tool(self, server, *, tool_name, arguments):
                raise AssertionError("unavailable MCP must not be called")

        client = MCPRegistryClient(
            servers=(MCPServerConfig(id="chrome", tool_prefix="chrome", command="node", allow_tools=("list_pages",)),),
            transport=FakeTransport(),
        )

        client.discover_tools()

        self.assertEqual(["chrome.list_pages"], client.configured_tool_names())
        self.assertEqual("unavailable", client.statuses()[0]["status"])
        result = client.call_tool(tool_name="chrome.list_pages", arguments={})
        self.assertEqual("MCP_TOOL_NOT_DISCOVERED", result.error)

    def test_declared_tool_names_remain_available_for_child_agent_registration_after_failed_discovery(self) -> None:
        from app.mcp_gateway.configured import configured_mcp_tool_names
        from app.mcp_gateway.registry import MCPRegistryClient
        from app.mcp_gateway.registry_config import MCPServerConfig

        class FakeTransport:
            async def list_tools(self, server):
                return []

            async def call_tool(self, server, *, tool_name, arguments):
                raise AssertionError("unavailable MCP must not reach transport")

        client = MCPRegistryClient(
            servers=(
                MCPServerConfig(
                    id="dbx",
                    tool_prefix="dbx",
                    command="node",
                    allow_tools=("dbx_list_connections",),
                ),
            ),
            transport=FakeTransport(),
        )

        client.discover_tools()

        self.assertEqual(["dbx.dbx_list_connections"], configured_mcp_tool_names(client))

    def test_registry_enforces_dbx_read_only_before_transport(self) -> None:
        from app.mcp_gateway.registry import MCPRegistryClient
        from app.mcp_gateway.registry_config import MCPServerConfig

        calls = []

        class FakeTransport:
            async def list_tools(self, server):
                return [{"name": "dbx_execute_query", "inputSchema": {"type": "object"}}]

            async def call_tool(self, server, *, tool_name, arguments):
                calls.append(arguments)
                raise AssertionError("write SQL must be blocked before transport")

        client = MCPRegistryClient(
            servers=(
                MCPServerConfig(
                    id="dbx",
                    tool_prefix="dbx",
                    command="node",
                    allow_tools=("dbx_execute_query",),
                    policy="dbx_read_only",
                ),
            ),
            transport=FakeTransport(),
        )
        client.discover_tools()

        result = client.call_tool(tool_name="dbx.dbx_execute_query", arguments={"sql": "DELETE FROM companies"})

        self.assertFalse(result.ok)
        self.assertEqual("DBX_READ_ONLY_POLICY", result.error)
        self.assertEqual([], calls)

    def test_mcp_definitions_use_discovered_schema(self) -> None:
        from app.agent_runtime.tool_registry import create_mcp_agent_tool_definitions
        from app.mcp_gateway.registry import MCPRegistryClient
        from app.mcp_gateway.registry_config import MCPServerConfig

        class FakeTransport:
            async def list_tools(self, server):
                return [{"name": "list_pages", "description": "List browser pages", "inputSchema": {"type": "object", "required": ["active"]}}]

            async def call_tool(self, server, *, tool_name, arguments):
                from app.mcp_gateway.client import MCPToolCallResult

                return MCPToolCallResult(tool_name=tool_name, ok=True, result={"pages": []})

        client = MCPRegistryClient(
            servers=(MCPServerConfig(id="chrome", tool_prefix="chrome", command="node", allow_tools=("list_pages",)),),
            transport=FakeTransport(),
        )
        client.discover_tools()
        definitions = create_mcp_agent_tool_definitions(client, allowed_tool_names=client.registered_tool_names())

        self.assertEqual({"active"}, set(definitions[0].input_schema["required"]))
        self.assertEqual("List browser pages", definitions[0].description)


if __name__ == "__main__":
    unittest.main()
