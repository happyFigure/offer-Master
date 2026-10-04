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
        self.assertIn("AgentRuntimeMcpIntegration", types_source)
        self.assertIn("mcp_integrations", types_source)
        self.assertIn('"configured"', types_source)
        self.assertIn("configured_tools", types_source)
        self.assertIn("discovered_tools", types_source)
        self.assertIn('"offline"', types_source)
        self.assertIn("candidate_use_when", types_source)
        self.assertIn("candidate_do_not_use_when", types_source)
        self.assertIn("candidate_positive_examples", types_source)
        self.assertIn("candidate_negative_examples", types_source)
        self.assertIn("kind:", types_source)

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
        self.assertIn("适用", app_source)
        self.assertIn("不适用", app_source)
        self.assertIn("capability.candidate_use_when", app_source)
        self.assertIn("capability.candidate_do_not_use_when", app_source)
        self.assertIn("Skill 执行", app_source)
        self.assertIn("McpIntegrationList", app_source)
        self.assertIn("MCP 接入状态", app_source)
        self.assertIn("panel.mcp_integrations", app_source)
        self.assertIn("mcp-integration-card", app_source)
        self.assertIn("声明工具", app_source)
        self.assertIn("实际发现工具", app_source)
        self.assertIn("runtimeEventRunsInSkill", app_source)
        self.assertIn("executorId.includes(\"openai\") || executorId.includes(\"claude\")", app_source)

    def test_task_finished_tone_checks_status_before_showing_success(self) -> None:
        app_source = Path("apps/web/src/app/App.tsx").read_text(encoding="utf-8")

        self.assertIn('if (eventType === "task_finished" && (status === "succeeded" || status === "success"))', app_source)
        self.assertIn('if (eventType === "task_finished" && (status === "failed" || status === "blocked"))', app_source)
        self.assertNotIn('if (eventType === "task_finished" || status === "succeeded" || status === "success")', app_source)

    def test_stale_approval_errors_clear_pending_card_and_resync_chat(self) -> None:
        app_source = Path("apps/web/src/app/App.tsx").read_text(encoding="utf-8")
        client_source = Path("apps/web/src/api/client.ts").read_text(encoding="utf-8")

        self.assertIn("resyncAgentSessionAfterApprovalStateChange", app_source)
        self.assertIn("isStaleApprovalError", app_source)
        self.assertIn('"STALE_APPROVAL_REQUEST"', app_source)
        self.assertIn("setPendingApproval(null)", app_source)
        self.assertIn("这个确认请求已经过期，已刷新当前会话状态，请重新发起操作。", app_source)
        self.assertIn('typeof nestedDetail.message === "string"', client_source)

    def test_runtime_timeline_renders_filesystem_postcheck_card(self) -> None:
        app_source = Path("apps/web/src/app/App.tsx").read_text(encoding="utf-8")
        css_source = Path("apps/web/src/styles/global.css").read_text(encoding="utf-8")

        self.assertIn("RuntimeFilesystemTraceCard", app_source)
        self.assertIn("runtimeFilesystemTrace", app_source)
        self.assertIn("文件复核", app_source)
        self.assertIn("source_exists_after", app_source)
        self.assertIn("target_exists_after", app_source)
        self.assertIn("runtime-filesystem-trace-card", css_source)

    def test_runtime_timeline_surfaces_unfinished_tool_calls(self) -> None:
        app_source = Path("apps/web/src/app/App.tsx").read_text(encoding="utf-8")
        css_source = Path("apps/web/src/styles/global.css").read_text(encoding="utf-8")

        self.assertIn("runtimeUnfinishedToolCalls", app_source)
        self.assertIn("RuntimeUnfinishedToolCallNotice", app_source)
        self.assertIn("工具结果未返回", app_source)
        self.assertIn("不要把流程结束当成工具成功", app_source)
        self.assertIn("runtime-unfinished-tool-notice", css_source)

    def test_runtime_timeline_surfaces_real_child_agent_delegations(self) -> None:
        app_source = Path("apps/web/src/app/App.tsx").read_text(encoding="utf-8")
        delegation_source = Path("apps/web/src/app/runtimeDelegations.ts").read_text(encoding="utf-8")
        agent_types_source = Path("apps/web/src/types/agent.ts").read_text(encoding="utf-8")
        css_source = Path("apps/web/src/styles/global.css").read_text(encoding="utf-8")

        self.assertIn('"subagent_started"', app_source)
        self.assertIn('"subagent_finished"', app_source)
        self.assertIn("delegationId", app_source)
        self.assertIn("RuntimeSubAgentSummary", app_source)
        self.assertIn("summarizeRuntimeDelegations", app_source)
        self.assertIn("mcpCallCount", delegation_source)
        self.assertIn("delegation_id", agent_types_source)
        self.assertIn("subagent_name", agent_types_source)
        self.assertIn("runtime-subagent-summary", css_source)
        self.assertIn("runtime-subagent-run", css_source)

    def test_agent_console_css_matches_dense_dark_dashboard(self) -> None:
        css_source = Path("apps/web/src/styles/global.css").read_text(encoding="utf-8")

        self.assertIn(".agent-console-layout", css_source)
        self.assertIn(".agent-console-profile", css_source)
        self.assertIn(".agent-console-main", css_source)
        self.assertIn(".agent-member-card", css_source)
        self.assertIn(".agent-capability-card", css_source)
        self.assertIn(".mcp-integration-card", css_source)
        self.assertIn(".mcp-integration-status", css_source)
        self.assertIn(".agent-health-line", css_source)
        self.assertIn(".agent-status-offline", css_source)
        self.assertIn("overflow-wrap: anywhere;", css_source)
        self.assertIn("@media (max-width: 960px)", css_source)
