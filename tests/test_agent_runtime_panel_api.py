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
