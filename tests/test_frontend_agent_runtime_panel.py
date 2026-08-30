from pathlib import Path
from unittest import TestCase


class FrontendAgentRuntimePanelTest(TestCase):
    def test_agent_runtime_api_client_exposes_panel_endpoint(self) -> None:
        api_source = Path("apps/web/src/api/agentRuntime.ts").read_text(encoding="utf-8")
        types_source = Path("apps/web/src/types/agentRuntime.ts").read_text(encoding="utf-8")

        self.assertIn("export async function getAgentRuntimePanel", api_source)
        self.assertIn('"/api/v1/agent-runtime/panel"', api_source)
        self.assertIn("export interface AgentRuntimePanel", types_source)
        self.assertIn("AgentRuntimeMember", types_source)
        self.assertIn("AgentRuntimeCapability", types_source)
        self.assertIn("AgentRuntimeHealth", types_source)
        self.assertIn('"offline"', types_source)

    def test_app_adds_agent_console_navigation_and_page(self) -> None:
        app_source = Path("apps/web/src/app/App.tsx").read_text(encoding="utf-8")

        self.assertIn('"agents"', app_source)
        self.assertIn("Agent 面板", app_source)
        self.assertIn("成员与能力注册", app_source)
        self.assertIn("getAgentRuntimePanel", app_source)
        self.assertIn("AgentRuntimePage", app_source)
        self.assertIn("agent-console-layout", app_source)
        self.assertIn("agent-member-card", app_source)
        self.assertIn("agent-capability-card", app_source)
        self.assertIn("能力声明", app_source)
        self.assertIn("agent.health", app_source)
        self.assertIn("未启动", app_source)
        self.assertIn("已连接", app_source)

    def test_agent_console_css_matches_dense_dark_dashboard(self) -> None:
        css_source = Path("apps/web/src/styles/global.css").read_text(encoding="utf-8")

        self.assertIn(".agent-console-layout", css_source)
        self.assertIn(".agent-console-profile", css_source)
        self.assertIn(".agent-console-main", css_source)
        self.assertIn(".agent-member-card", css_source)
        self.assertIn(".agent-capability-card", css_source)
        self.assertIn(".agent-health-line", css_source)
        self.assertIn(".agent-status-offline", css_source)
        self.assertIn("overflow-wrap: anywhere;", css_source)
        self.assertIn("@media (max-width: 960px)", css_source)
