from unittest import TestCase


class AgentToolRegistryTest(TestCase):
    def test_default_registry_registers_memory_and_content_source_tools_in_stable_order(self) -> None:
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry

        registry = create_default_agent_tool_registry()
        definitions = registry.list_definitions()

        self.assertEqual(
            [
                "memory_get",
                "memory_search",
                "sessions_history",
                "sessions_search",
                "weixin-articles-mcp.read_article",
                "xiaohongshu-mcp.get_feed_detail",
                "xiaohongshu-mcp.search_feeds",
            ],
            [definition.name for definition in definitions],
        )
        self.assertTrue(all(definition.input_schema for definition in definitions))
        self.assertTrue(all(definition.output_schema for definition in definitions))
        self.assertTrue(all(not definition.requires_confirmation for definition in definitions))

    def test_content_source_tool_definitions_call_their_adapter_methods(self) -> None:
        from app.agent_runtime.tool_registry import create_content_source_agent_tool_definitions
        from app.mcp_gateway.client import MCPToolCallResult

        class FakeContentSourceClient:
            def __init__(self) -> None:
                self.calls = []

            def read_weixin_article(self, *, url: str) -> MCPToolCallResult:
                self.calls.append(("read_weixin_article", {"url": url}))
                return MCPToolCallResult(
                    tool_name="weixin-articles-mcp.read_article",
                    ok=True,
                    result={"title": "Tencent 2027"},
                )

            def search_xiaohongshu_feeds(self, *, keyword: str, filters: dict | None = None) -> MCPToolCallResult:
                self.calls.append(("search_xiaohongshu_feeds", {"keyword": keyword, "filters": filters}))
                return MCPToolCallResult(
                    tool_name="xiaohongshu-mcp.search_feeds",
                    ok=True,
                    result={"items": [{"title": "2027 秋招"}]},
                )

            def get_xiaohongshu_feed_detail(self, **arguments) -> MCPToolCallResult:
                self.calls.append(("get_xiaohongshu_feed_detail", arguments))
                return MCPToolCallResult(
                    tool_name="xiaohongshu-mcp.get_feed_detail",
                    ok=True,
                    result={"feed_id": arguments["feed_id"], "text": "招聘信息"},
                )

        fake_client = FakeContentSourceClient()
        definitions = {definition.name: definition for definition in create_content_source_agent_tool_definitions(fake_client)}

        self.assertEqual(
            {"weixin-articles-mcp.read_article", "xiaohongshu-mcp.search_feeds", "xiaohongshu-mcp.get_feed_detail"},
            set(definitions),
        )

        wechat_result = definitions["weixin-articles-mcp.read_article"].handler(None, url="https://mp.weixin.qq.com/s/example")
        search_result = definitions["xiaohongshu-mcp.search_feeds"].handler(None, keyword="2027 秋招 Java")
        detail_result = definitions["xiaohongshu-mcp.get_feed_detail"].handler(
            None,
            feed_id="abc",
            xsec_token="token",
        )

        self.assertTrue(wechat_result.ok)
        self.assertTrue(search_result.ok)
        self.assertTrue(detail_result.ok)
        self.assertEqual(
            [
                ("read_weixin_article", {"url": "https://mp.weixin.qq.com/s/example"}),
                ("search_xiaohongshu_feeds", {"keyword": "2027 秋招 Java", "filters": None}),
                ("get_xiaohongshu_feed_detail", {"feed_id": "abc", "xsec_token": "token"}),
            ],
            fake_client.calls,
        )

    def test_content_source_client_delegates_xiaohongshu_tools_to_mcp_gateway(self) -> None:
        from app.mcp_gateway.client import MCPToolCallResult
        from app.mcp_gateway.content_source_client import ContentSourceMCPClient

        class FakeMCPGatewayClient:
            def __init__(self) -> None:
                self.calls = []

            def call_tool(self, *, tool_name: str, arguments: dict) -> MCPToolCallResult:
                self.calls.append({"tool_name": tool_name, "arguments": arguments})
                return MCPToolCallResult(tool_name=tool_name, ok=True, result={"ok": True})

        fake_gateway = FakeMCPGatewayClient()
        client = ContentSourceMCPClient(mcp_client=fake_gateway)

        search_result = client.search_xiaohongshu_feeds(keyword="2027 autumn recruit")
        detail_result = client.get_xiaohongshu_feed_detail(feed_id="abc", xsec_token="token")

        self.assertTrue(search_result.ok)
        self.assertTrue(detail_result.ok)
        self.assertEqual(
            [
                {"tool_name": "xiaohongshu-mcp.search_feeds", "arguments": {"keyword": "2027 autumn recruit", "filters": None}},
                {"tool_name": "xiaohongshu-mcp.get_feed_detail", "arguments": {"feed_id": "abc", "xsec_token": "token"}},
            ],
            fake_gateway.calls,
        )

    def test_content_source_client_calls_xiaohongshu_rest_api_when_base_url_configured(self) -> None:
        from app.mcp_gateway.content_source_client import ContentSourceMCPClient

        class FakeResponse:
            status_code = 200

            def __init__(self, payload: dict) -> None:
                self._payload = payload

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return self._payload

        class FakeHTTPClient:
            def __init__(self) -> None:
                self.calls = []

            def post(self, url: str, *, json: dict, headers: dict, timeout: float) -> FakeResponse:
                self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
                return FakeResponse({"success": True, "data": {"echo": json}, "message": "ok"})

        http_client = FakeHTTPClient()
        client = ContentSourceMCPClient(
            xiaohongshu_base_url="http://127.0.0.1:18060/",
            xiaohongshu_auth_token="secret-token",
            http_client=http_client,
        )

        search_result = client.search_xiaohongshu_feeds(keyword="2027 autumn recruit", filters={"sort_by": "latest"})
        detail_result = client.get_xiaohongshu_feed_detail(feed_id="abc", xsec_token="token", include_comments=True, comment_limit=20)

        self.assertTrue(search_result.ok)
        self.assertTrue(detail_result.ok)
        self.assertEqual(
            [
                {
                    "url": "http://127.0.0.1:18060/api/v1/feeds/search",
                    "json": {"keyword": "2027 autumn recruit", "filters": {"sort_by": "latest"}},
                    "headers": {"Authorization": "Bearer secret-token"},
                    "timeout": 30.0,
                },
                {
                    "url": "http://127.0.0.1:18060/api/v1/feeds/detail",
                    "json": {"feed_id": "abc", "xsec_token": "token", "load_all_comments": True, "limit": 20},
                    "headers": {"Authorization": "Bearer secret-token"},
                    "timeout": 30.0,
                },
            ],
            http_client.calls,
        )

    def test_guard_blocks_unregistered_tool_with_structured_error(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolRuntimeGuard
        from app.agent_runtime.tool_registry import AgentToolRegistry

        result = AgentToolRuntimeGuard().pre_check(
            AgentToolCallContext(stage="plan", tool_name="unknown_tool", source_type="agent_chat"),
            registry=AgentToolRegistry(),
        )

        self.assertFalse(result.ok)
        self.assertEqual("TOOL_NOT_REGISTERED", result.error_code)
        self.assertEqual("plan", result.stage)
        self.assertEqual("unknown_tool", result.tool_name)
        self.assertEqual("stop", result.next_action)
        self.assertIn("unknown_tool", result.reason)
        self.assertIn("not registered", result.user_message)

    def test_guard_blocks_tool_budget_exceeded(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolPolicy, AgentToolRuntimeGuard
        from app.agent_runtime.tool_registry import create_default_agent_tool_registry

        result = AgentToolRuntimeGuard(policy=AgentToolPolicy(max_tool_calls=10)).pre_check(
            AgentToolCallContext(
                stage="recall",
                tool_name="sessions_search",
                source_type="agent_chat",
                tool_call_count=10,
            ),
            registry=create_default_agent_tool_registry(),
        )

        self.assertFalse(result.ok)
        self.assertEqual("TOOL_BUDGET_EXCEEDED", result.error_code)
        self.assertEqual("stop", result.next_action)
        self.assertEqual({"tool_calls": 10, "max_tool_calls": 10}, result.cost)
        self.assertEqual("max_tool_calls", result.error_details["budget_name"])

    def test_guard_blocks_source_type_not_allowed(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolRuntimeGuard
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry

        registry = AgentToolRegistry()
        registry.register(
            AgentToolDefinition(
                name="wechat_visible_page",
                description="Read a user-visible WeChat page.",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                allowed_source_types=frozenset({"wechat_article"}),
            )
        )

        result = AgentToolRuntimeGuard().pre_check(
            AgentToolCallContext(
                stage="fetch",
                tool_name="wechat_visible_page",
                source_type="xiaohongshu_note",
            ),
            registry=registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual("TOOL_SOURCE_TYPE_NOT_ALLOWED", result.error_code)
        self.assertEqual("select_alternative_tool", result.next_action)
        self.assertEqual("xiaohongshu_note", result.error_details["source_type"])
        self.assertEqual(["wechat_article"], result.error_details["allowed_source_types"])

    def test_guard_requires_confirmation_for_high_risk_tool(self) -> None:
        from app.agent_runtime.guardrails import AgentToolCallContext, AgentToolRuntimeGuard
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel

        registry = AgentToolRegistry()
        registry.register(
            AgentToolDefinition(
                name="submit_application",
                description="Submit a real job application.",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                risk_level=AgentToolRiskLevel.HIGH,
                requires_confirmation=True,
                allowed_source_types=frozenset({"application"}),
            )
        )

        blocked = AgentToolRuntimeGuard().pre_check(
            AgentToolCallContext(
                stage="apply",
                tool_name="submit_application",
                source_type="application",
                user_confirmed=False,
            ),
            registry=registry,
        )
        allowed = AgentToolRuntimeGuard().pre_check(
            AgentToolCallContext(
                stage="apply",
                tool_name="submit_application",
                source_type="application",
                user_confirmed=True,
            ),
            registry=registry,
        )

        self.assertFalse(blocked.ok)
        self.assertEqual("TOOL_USER_CONFIRMATION_REQUIRED", blocked.error_code)
        self.assertEqual("request_user_confirmation", blocked.next_action)
        self.assertTrue(allowed.ok)
        self.assertEqual("continue", allowed.next_action)

    def test_mcp_tool_policy_normalizes_allowlist_and_confirmation_boundary(self) -> None:
        from app.mcp_gateway.tool_policy import MCPToolPolicy

        policy = MCPToolPolicy.from_allowlist([" open_page ", "read_page", "fill_form", ""])

        self.assertEqual(["open_page", "read_page", "fill_form"], policy.allowed_tool_names())
        self.assertTrue(policy.is_allowed("open_page"))
        self.assertFalse(policy.is_allowed("unknown_tool"))
        self.assertFalse(policy.requires_confirmation("open_page"))
        self.assertTrue(policy.requires_confirmation("fill_form"))
