import anyio
import unittest
from unittest.mock import patch


class StdioMCPGatewayClientTests(unittest.TestCase):
    def test_routes_prefixed_tool_name_to_matching_stdio_server(self) -> None:
        from app.mcp_gateway.client import MCPToolCallResult
        from app.mcp_gateway.stdio_client import MCPStdioServerSpec, StdioMCPGatewayClient

        calls = []

        class FakeTransport:
            async def call_tool(self, server, *, tool_name, arguments):
                calls.append({"prefix": server.tool_prefix, "tool_name": tool_name, "arguments": arguments})
                return MCPToolCallResult(tool_name=tool_name, ok=True, result={"items": []})

        client = StdioMCPGatewayClient(
            servers=(
                MCPStdioServerSpec(tool_prefix="chrome", command="cmd", args=("/c", "chrome-mcp")),
                MCPStdioServerSpec(tool_prefix="dbx", command="cmd", args=("/c", "dbx-mcp")),
            ),
            transport=FakeTransport(),
        )

        result = client.call_tool(tool_name="dbx.dbx_list_connections", arguments={"limit": 5})

        self.assertTrue(result.ok)
        self.assertEqual(
            [{"prefix": "dbx", "tool_name": "dbx_list_connections", "arguments": {"limit": 5}}],
            calls,
        )

    def test_sync_call_tool_is_safe_inside_existing_event_loop(self) -> None:
        from app.mcp_gateway.client import MCPToolCallResult
        from app.mcp_gateway.stdio_client import MCPStdioServerSpec, StdioMCPGatewayClient

        class FakeTransport:
            async def call_tool(self, server, *, tool_name, arguments):
                return MCPToolCallResult(tool_name=tool_name, ok=True, result={"tool": tool_name})

        client = StdioMCPGatewayClient(
            servers=(MCPStdioServerSpec(tool_prefix="chrome", command="cmd", args=("/c", "chrome-mcp")),),
            transport=FakeTransport(),
        )

        async def run_inside_loop():
            return client.call_tool(tool_name="chrome.list_pages", arguments={})

        result = anyio.run(run_inside_loop)

        self.assertTrue(result.ok)
        self.assertEqual({"tool": "list_pages"}, result.result)

    def test_unknown_prefixed_tool_returns_structured_error(self) -> None:
        from app.mcp_gateway.stdio_client import MCPStdioServerSpec, StdioMCPGatewayClient

        client = StdioMCPGatewayClient(
            servers=(MCPStdioServerSpec(tool_prefix="chrome", command="cmd", args=("/c", "chrome-mcp")),)
        )

        result = client.call_tool(tool_name="dbx.dbx_list_connections", arguments={})

        self.assertFalse(result.ok)
        self.assertEqual("MCP_SERVER_NOT_CONFIGURED", result.error)

    def test_persistent_transport_reuses_one_session_for_discovery_and_calls(self) -> None:
        from app.mcp_gateway.client import MCPToolCallResult
        from app.mcp_gateway.stdio_client import MCPStdioServerSpec, PersistentStdioMCPTransport

        created = []

        class FakeSession:
            def __init__(self, server):
                self.server = server
                created.append(self)

            def submit(self, operation, **payload):
                if operation == "list_tools":
                    return {"tools": [{"name": "list_pages", "inputSchema": {"type": "object"}}]}
                return {"content": [{"type": "text", "text": payload["tool_name"]}]}

            def close(self):
                return None

        async def exercise():
            transport = PersistentStdioMCPTransport()
            server = MCPStdioServerSpec(tool_prefix="chrome", command="cmd", args=("/c", "chrome-mcp"))
            discovered = await transport.list_tools(server)
            result = await transport.call_tool(server, tool_name="list_pages", arguments={})
            transport.close()
            return discovered, result

        with patch("app.mcp_gateway.stdio_client._PersistentServerSession", FakeSession):
            discovered, result = anyio.run(exercise)

        self.assertEqual([{"name": "list_pages", "inputSchema": {"type": "object"}}], discovered)
        self.assertIsInstance(result, MCPToolCallResult)
        self.assertTrue(result.ok)
        self.assertEqual(1, len(created))


if __name__ == "__main__":
    unittest.main()
