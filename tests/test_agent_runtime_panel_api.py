import sys
from asyncio import run
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class AgentRuntimePanelApiTest(TestCase):
    def test_runtime_panel_exposes_main_agent_members_and_capabilities(self) -> None:
        from app.main import create_app

        app = create_app()

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/api/v1/agent-runtime/panel")

        response = run(call_api())

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("offermaster-main-agent", payload["main_agent"]["id"])
        self.assertIn("负责会话", payload["main_agent"]["description"])
        self.assertGreaterEqual(payload["summary"]["agent_count"], 1)
        self.assertGreaterEqual(payload["summary"]["capability_count"], 1)
        self.assertIn("agents", payload)
        self.assertIn("capabilities", payload)
        self.assertTrue(any(agent["id"] == "agent_tool_registry" for agent in payload["agents"]))
        self.assertTrue(any(capability["id"] == "external.web_search" for capability in payload["capabilities"]))
        web_search = next(capability for capability in payload["capabilities"] if capability["id"] == "external.web_search")
        self.assertEqual("low", web_search["risk_level"])
        self.assertFalse(web_search["requires_confirmation"])
        self.assertIn("query", web_search["input_fields"])
        self.assertIn("agent_chat", web_search["allowed_source_types"])
        self.assertIn("public_web_information", web_search["candidate_categories"])

    def test_runtime_panel_marks_configured_claude_sdk_agent_offline_when_heartbeat_fails(self) -> None:
        from app.api.v1.agent_runtime import get_settings
        from app.core.config import Settings
        from app.main import create_app

        app = create_app()
        app.dependency_overrides[get_settings] = lambda: Settings(
            external_agent_auto_dispatch=True,
            external_web_search_provider="bailian",
            claude_sdk_agent_base_url="http://127.0.0.1:65535",
        )

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/api/v1/agent-runtime/panel")

        response = run(call_api())

        self.assertEqual(200, response.status_code)
        claude_agent = next(agent for agent in response.json()["agents"] if agent["id"] == "claude-sdk-agent")
        self.assertEqual("offline", claude_agent["status"])
        self.assertEqual("unreachable", claude_agent["health"]["status"])
        self.assertIn("未启动", claude_agent["health"]["label"])

    def test_runtime_panel_marks_configured_claude_sdk_agent_standby_when_heartbeat_succeeds_but_provider_is_bailian(self) -> None:
        from app.api.v1.agent_runtime import get_settings
        from app.core.config import Settings
        from app.main import create_app

        app = create_app()
        app.dependency_overrides[get_settings] = lambda: Settings(
            external_agent_auto_dispatch=True,
            external_web_search_provider="bailian",
            claude_sdk_agent_base_url="http://claude-agent.test",
        )

        class FakeResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/api/v1/agent-runtime/panel")

        with patch("app.api.v1.agent_runtime.httpx.get", return_value=FakeResponse()):
            response = run(call_api())

        self.assertEqual(200, response.status_code)
        claude_agent = next(agent for agent in response.json()["agents"] if agent["id"] == "claude-sdk-agent")
        self.assertEqual("standby", claude_agent["status"])
        self.assertEqual("healthy", claude_agent["health"]["status"])
        self.assertIn("已连接", claude_agent["health"]["label"])

    def test_runtime_panel_shows_openai_sdk_agent_as_not_configured_when_disabled(self) -> None:
        from app.api.v1.agent_runtime import get_settings
        from app.core.config import Settings
        from app.main import create_app

        app = create_app()
        app.dependency_overrides[get_settings] = lambda: Settings(
            external_agent_auto_dispatch=True,
            openai_sdk_agent_enabled=False,
            openai_sdk_agent_api_key=None,
        )

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/api/v1/agent-runtime/panel")

        response = run(call_api())

        self.assertEqual(200, response.status_code)
        openai_agent = next(agent for agent in response.json()["agents"] if agent["id"] == "openai-sdk-agent")
        self.assertEqual("offline", openai_agent["status"])
        self.assertEqual("not_configured", openai_agent["health"]["status"])
        self.assertIn("未配置", openai_agent["health"]["label"])
        self.assertEqual([], openai_agent["capabilities"])

    def test_runtime_panel_marks_openai_sdk_agent_configured_from_main_llm_settings(self) -> None:
        from app.api.v1.agent_runtime import get_settings
        from app.core.config import Settings
        from app.main import create_app

        app = create_app()
        app.dependency_overrides[get_settings] = lambda: Settings(
            external_agent_auto_dispatch=True,
            external_web_search_provider="bailian",
            openai_sdk_agent_enabled=True,
            openai_sdk_agent_api_key=None,
            openai_sdk_agent_model=None,
            openai_sdk_agent_base_url=None,
            llm_api_key="sk-main-model",
            llm_model="qwen-plus",
            llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/api/v1/agent-runtime/panel")

        response = run(call_api())

        self.assertEqual(200, response.status_code)
        openai_agent = next(agent for agent in response.json()["agents"] if agent["id"] == "openai-sdk-agent")
        self.assertEqual("active", openai_agent["status"])
        self.assertEqual("healthy", openai_agent["health"]["status"])
        self.assertIn("已配置", openai_agent["health"]["label"])
        self.assertTrue(any(capability["id"] == "resume.tailor" for capability in openai_agent["capabilities"]))

    def test_runtime_panel_binds_filesystem_skill_to_openai_sdk_agent_when_delegated(self) -> None:
        from app.api.v1.agent_runtime import get_settings
        from app.core.config import Settings
        from app.main import create_app

        app = create_app()
        app.dependency_overrides[get_settings] = lambda: Settings(
            external_agent_auto_dispatch=True,
            external_web_search_provider="bailian",
            openai_sdk_agent_enabled=True,
            openai_sdk_agent_mode="agents_sdk",
            openai_sdk_agent_api_key=None,
            openai_sdk_agent_model=None,
            openai_sdk_agent_base_url=None,
            llm_api_key="sk-main-model",
            llm_model="qwen-plus",
            llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            sdk_agent_enable_file_analysis=True,
            sdk_agent_enable_mutation_tools=True,
            sdk_agent_sandbox_mode="temp_copy",
        )

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/api/v1/agent-runtime/panel")

        response = run(call_api())

        self.assertEqual(200, response.status_code)
        payload = response.json()
        filesystem_skill = next(capability for capability in payload["capabilities"] if capability["id"] == "skill.filesystem")
        openai_agent = next(agent for agent in payload["agents"] if agent["id"] == "openai-sdk-agent")
        filesystem_agent = next((agent for agent in payload["agents"] if agent["id"] == "skill_executor.filesystem"), None)

        self.assertEqual("openai-sdk-agent", filesystem_skill["executor_id"])
        self.assertTrue(any(capability["id"] == "skill.filesystem" for capability in openai_agent["capabilities"]))
        self.assertIsNone(filesystem_agent)

    def test_runtime_panel_registers_configured_mcp_tools_for_sdk_child_agent(self) -> None:
        from app.api.v1.agent_runtime import get_settings
        from app.core.config import Settings
        from app.main import create_app

        app = create_app()
        app.dependency_overrides[get_settings] = lambda: Settings(
            external_agent_auto_dispatch=True,
            external_web_search_provider="bailian",
            openai_sdk_agent_enabled=True,
            openai_sdk_agent_mode="agents_sdk",
            openai_sdk_agent_api_key=None,
            openai_sdk_agent_model=None,
            openai_sdk_agent_base_url=None,
            llm_api_key="sk-main-model",
            llm_model="qwen-plus",
            llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            mcp_enabled=True,
            mcp_server_url="http://127.0.0.1:18080",
            sdk_agent_enable_chrome_mcp=True,
            sdk_agent_enable_dbx_mcp=True,
        )

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/api/v1/agent-runtime/panel")

        response = run(call_api())

        self.assertEqual(200, response.status_code)
        payload = response.json()
        capabilities_by_id = {capability["id"]: capability for capability in payload["capabilities"]}
        integrations_by_id = {integration["id"]: integration for integration in payload["mcp_integrations"]}

        self.assertEqual("openai-sdk-agent", capabilities_by_id["mcp.chrome.list_pages"]["executor_id"])
        self.assertEqual("openai-sdk-agent", capabilities_by_id["mcp.dbx.dbx_list_connections"]["executor_id"])
        self.assertEqual("registered", integrations_by_id["chrome"]["status"])
        self.assertEqual("registered", integrations_by_id["dbx"]["status"])
        self.assertIn("mcp.chrome.list_pages", integrations_by_id["chrome"]["registered_tools"])
        self.assertIn("mcp.dbx.dbx_list_connections", integrations_by_id["dbx"]["registered_tools"])

    def test_runtime_panel_shows_a_separate_openai_sdk_mail_agent(self) -> None:
        from app.api.v1.agent_runtime import get_settings
        from app.agent_runtime.agent_as_tool import OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID
        from app.core.config import Settings
        from app.main import create_app

        app = create_app()
        app.dependency_overrides[get_settings] = lambda: Settings(
            external_agent_auto_dispatch=True,
            external_web_search_provider="bailian",
            openai_sdk_agent_enabled=True,
            openai_sdk_agent_mode="agents_sdk",
            openai_sdk_agent_api_key="test-api-key",
            openai_sdk_agent_model="test-model",
            sdk_agent_enable_qq_mail_mcp=True,
            qq_mail_username="candidate@example.com",
            qq_mail_auth_code="test-auth-code",
        )

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/api/v1/agent-runtime/panel")

        response = run(call_api())

        self.assertEqual(200, response.status_code)
        payload = response.json()
        agents_by_id = {agent["id"]: agent for agent in payload["agents"]}
        capabilities_by_id = {capability["id"]: capability for capability in payload["capabilities"]}
        mail_agent = agents_by_id[OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID]

        self.assertEqual("OpenAI SDK Agent（QQ 邮箱）", mail_agent["name"])
        self.assertTrue(any(capability["id"] == "agent.qq_mail_readonly" for capability in mail_agent["capabilities"]))
        self.assertEqual(OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID, capabilities_by_id["agent.qq_mail_readonly"]["executor_id"])
        self.assertEqual(OPENAI_SDK_QQ_MAIL_AGENT_EXECUTOR_ID, capabilities_by_id["mcp.qq_mail.list_messages"]["executor_id"])
        general_agent = agents_by_id["openai-sdk-agent"]
        self.assertFalse(any(capability["id"] == "agent.qq_mail_readonly" for capability in general_agent["capabilities"]))

    def test_runtime_panel_uses_default_stdio_mcp_bridge_without_http_gateway(self) -> None:
        from app.api.v1.agent_runtime import get_settings
        from app.core.config import Settings
        from app.main import create_app

        app = create_app()
        app.dependency_overrides[get_settings] = lambda: Settings(
            external_agent_auto_dispatch=True,
            external_web_search_provider="bailian",
            openai_sdk_agent_enabled=True,
            openai_sdk_agent_mode="agents_sdk",
            openai_sdk_agent_api_key=None,
            openai_sdk_agent_model=None,
            openai_sdk_agent_base_url=None,
            llm_api_key="sk-main-model",
            llm_model="qwen-plus",
            llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            mcp_enabled=False,
            mcp_server_url=None,
            sdk_agent_enable_chrome_mcp=True,
            sdk_agent_enable_dbx_mcp=True,
        )

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/api/v1/agent-runtime/panel")

        response = run(call_api())

        self.assertEqual(200, response.status_code)
        integrations_by_id = {integration["id"]: integration for integration in response.json()["mcp_integrations"]}

        self.assertEqual("configured", integrations_by_id["chrome"]["status"])
        self.assertEqual("configured", integrations_by_id["dbx"]["status"])
        self.assertIn("SDK", integrations_by_id["chrome"]["detail"])
        self.assertIn("mcp.chrome.list_pages", integrations_by_id["chrome"]["registered_tools"])
        self.assertIn("mcp.dbx.dbx_list_connections", integrations_by_id["dbx"]["registered_tools"])

    def test_runtime_panel_does_not_discover_stdio_mcp_on_cold_start(self) -> None:
        from app.api.v1.agent_runtime import get_settings
        from app.core.config import Settings
        from app.main import create_app

        app = create_app()
        app.dependency_overrides[get_settings] = lambda: Settings(
            external_agent_auto_dispatch=True,
            external_web_search_provider="bailian",
            openai_sdk_agent_enabled=True,
            openai_sdk_agent_mode="agents_sdk",
            openai_sdk_agent_api_key=None,
            openai_sdk_agent_model=None,
            openai_sdk_agent_base_url=None,
            llm_api_key="sk-main-model",
            llm_model="qwen-plus",
            llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            mcp_enabled=False,
            mcp_server_url=None,
            sdk_agent_enable_chrome_mcp=True,
            sdk_agent_enable_dbx_mcp=True,
        )

        async def call_api():
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get("/api/v1/agent-runtime/panel")

        with patch(
            "app.mcp_gateway.registry.MCPRegistryClient.discover_tools",
            side_effect=AssertionError("panel must not start MCP discovery"),
        ):
            response = run(call_api())

        self.assertEqual(200, response.status_code)
        integrations_by_id = {integration["id"]: integration for integration in response.json()["mcp_integrations"]}
        self.assertEqual("configured", integrations_by_id["chrome"]["status"])
        self.assertEqual("configured", integrations_by_id["dbx"]["status"])
        self.assertIn("SDK", integrations_by_id["chrome"]["detail"])
        self.assertEqual([], integrations_by_id["chrome"]["discovered_tools"])
