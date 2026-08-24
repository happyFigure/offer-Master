from pathlib import Path
from unittest import TestCase


class FrontendAgentChatApiTest(TestCase):
    def test_agent_api_client_uses_session_message_and_stream_endpoints(self) -> None:
        api_source = Path("apps/web/src/api/agent.ts").read_text(encoding="utf-8")

        self.assertIn("export async function createAgentSession", api_source)
        self.assertIn("export async function listAgentSessions", api_source)
        self.assertIn("export async function getAgentMessages", api_source)
        self.assertIn("export async function updateAgentSession", api_source)
        self.assertIn("export async function deleteAgentSession", api_source)
        self.assertIn("export async function sendAgentMessage", api_source)
        self.assertIn("export async function streamAgentMessage", api_source)
        self.assertIn("export async function approveAgentApproval", api_source)
        self.assertIn("export async function rejectAgentApproval", api_source)
        self.assertIn('"/api/v1/agent/sessions"', api_source)
        self.assertIn("PATCH", api_source)
        self.assertIn("DELETE", api_source)
        self.assertIn("`/api/v1/agent/sessions/${sessionId}/messages`", api_source)
        self.assertIn("/api/v1/agent/sessions/${sessionId}/messages/stream", api_source)
        self.assertIn('eventName === "approval_required"', api_source)
        self.assertIn('eventName === "outer_session_event"', api_source)
        self.assertIn('eventName === "tool_event"', api_source)
        self.assertIn("onApprovalRequired", api_source)
        self.assertIn("onOuterSessionEvent", api_source)
        self.assertIn("onToolEvent", api_source)

    def test_vite_dev_server_proxies_api_requests_to_fastapi(self) -> None:
        vite_config = Path("apps/web/vite.config.ts").read_text(encoding="utf-8")

        self.assertIn("proxy", vite_config)
        self.assertIn('"/api"', vite_config)
        self.assertIn('target: "http://127.0.0.1:8000"', vite_config)
        self.assertIn("changeOrigin: true", vite_config)

    def test_agent_types_expose_context_metadata_for_frontend_status(self) -> None:
        types_source = Path("apps/web/src/types/agent.ts").read_text(encoding="utf-8")

        self.assertIn("export interface AgentSession", types_source)
        self.assertIn("export interface AgentSessionUpdateInput", types_source)
        self.assertIn("export interface AgentMessage", types_source)
        self.assertIn("export interface AgentContextSummary", types_source)
        self.assertIn("export interface AgentCompactResult", types_source)
        self.assertIn("export interface AgentApprovalRequest", types_source)
        self.assertIn("export interface AgentApprovalRequiredPayload", types_source)
        self.assertIn("export interface AgentApprovalDecisionResponse", types_source)
        self.assertIn("export interface AgentStreamOuterSessionEvent", types_source)
        self.assertIn("export interface AgentStreamToolEvent", types_source)
        self.assertIn("requested_tool_name", types_source)
        self.assertIn("user_confirmed", types_source)
        self.assertIn("export interface AgentContextMetadata", types_source)
        self.assertIn("loaded_session_history_ids", types_source)
        self.assertIn("need_compaction", types_source)
        self.assertIn("auto_compacted", types_source)

    def test_chat_page_streams_agent_reply_and_does_not_show_manual_compact_action(self) -> None:
        app_source = Path("apps/web/src/app/App.tsx").read_text(encoding="utf-8")

        self.assertIn("loadAgentSession", app_source)
        self.assertIn("agentSessions", app_source)
        self.assertIn("handleCreateChatSession", app_source)
        self.assertIn("handleRenameChatSession", app_source)
        self.assertIn("handleDeleteChatSession", app_source)
        self.assertIn("onSelectSession", app_source)
        self.assertIn("streamAgentMessage", app_source)
        self.assertIn("handleChatInputKeyDown", app_source)
        self.assertIn("pendingApproval", app_source)
        self.assertIn("handleApprovePendingApproval", app_source)
        self.assertIn("handleRejectPendingApproval", app_source)
        self.assertIn("ToolApprovalCard", app_source)
        self.assertIn("runtimeEvents", app_source)
        self.assertIn("ChatRuntimeTimeline", app_source)
        self.assertIn("appendRuntimeEvent", app_source)
        self.assertIn("formatRuntimeEventSummary", app_source)
        self.assertIn("执行过程", app_source)
        self.assertIn('"local.company_database_overview": "本地企业库概览"', app_source)
        self.assertIn('event.toolName?.includes("database")', app_source)
        self.assertIn("onOuterSessionEvent", app_source)
        self.assertIn("onToolEvent", app_source)
        self.assertIn('event.key === "Enter"', app_source)
        self.assertIn("event.shiftKey", app_source)
        self.assertIn("contextMetadata", app_source)
        self.assertNotIn("onCompactSession", app_source)
        self.assertNotIn("compactAgentSession", app_source)
        self.assertNotIn("手动压缩会话", app_source)
        self.assertNotIn("assistant-placeholder", app_source)

    def test_chat_page_renders_markdown_tables_as_table_cards(self) -> None:
        app_source = Path("apps/web/src/app/App.tsx").read_text(encoding="utf-8")
        style_source = Path("apps/web/src/styles/global.css").read_text(encoding="utf-8")

        self.assertIn("ChatMessageContent", app_source)
        self.assertIn("ChatTableCard", app_source)
        self.assertIn("parseMarkdownTables", app_source)
        self.assertIn("chat-table-card", app_source)
        self.assertIn('aria-label="复制表格"', app_source)
        self.assertIn('aria-label="下载表格"', app_source)
        self.assertIn(".chat-table-card", style_source)
        self.assertIn(".chat-table-card table", style_source)
        self.assertIn("overflow-x: auto", style_source)

    def test_approval_buttons_merge_returned_assistant_message_into_chat(self) -> None:
        app_source = Path("apps/web/src/app/App.tsx").read_text(encoding="utf-8")

        self.assertIn("appendAgentMessageIfMissing", app_source)
        self.assertIn("appendAgentMessageIfMissing(messages, result.assistant_message)", app_source)
        self.assertIn("setPendingApproval(null)", app_source)

    def test_runtime_timeline_spins_only_latest_running_event_while_working(self) -> None:
        app_source = Path("apps/web/src/app/App.tsx").read_text(encoding="utf-8")

        self.assertIn("runtimeEventShouldSpin", app_source)
        self.assertIn("latestEvent", app_source)
        self.assertIn('isWorking && event.tone === "running"', app_source)
        self.assertNotIn('event.tone === "running" ? "spin-slow"', app_source)

    def test_runtime_timeline_displays_loop_choice_events_in_plain_chinese(self) -> None:
        app_source = Path("apps/web/src/app/App.tsx").read_text(encoding="utf-8")
        api_source = Path("apps/web/src/api/agent.ts").read_text(encoding="utf-8")
        types_source = Path("apps/web/src/types/agent.ts").read_text(encoding="utf-8")

        self.assertIn('tool_name?: string | null', types_source)
        self.assertIn('capability?: string | null', types_source)
        self.assertIn('candidate_capabilities?: string[]', types_source)
        self.assertIn('"tool_name" in value || "capability" in value', api_source)
        self.assertIn('candidate_capabilities', app_source)
        self.assertIn('candidateNames', app_source)
        self.assertIn('候选能力', app_source)
        self.assertIn('模型选择能力', app_source)
        self.assertIn('开始思考', app_source)
        self.assertIn('观察结果', app_source)
        self.assertIn('结果不足，准备重试', app_source)
        self.assertIn('等待用户确认或补充', app_source)
        self.assertIn('runtime-event-candidates', app_source)

    def test_runtime_timeline_replaces_right_session_memory_panel(self) -> None:
        app_source = Path("apps/web/src/app/App.tsx").read_text(encoding="utf-8")

        side_panel_index = app_source.index('<aside className="glass-panel chat-side-panel"')
        timeline_index = app_source.index("<ChatRuntimeTimeline", side_panel_index)
        chat_panel_close_index = app_source.index('<aside className="glass-panel chat-side-panel"')

        self.assertGreaterEqual(timeline_index, side_panel_index)
        self.assertNotIn("Session Memory", app_source)
        self.assertNotIn("Auto Compaction", app_source)
        self.assertNotIn("会话记忆已连接", app_source)
        self.assertNotIn("边界不变", app_source)
        self.assertIn("Agent 执行过程", app_source)
        self.assertIn("暂无执行过程", app_source)
        self.assertNotIn("if (!events.length && !isWorking)", app_source)
        self.assertEqual(side_panel_index, chat_panel_close_index)

    def test_chat_page_auto_scrolls_only_when_user_stays_near_bottom(self) -> None:
        app_source = Path("apps/web/src/app/App.tsx").read_text(encoding="utf-8")

        self.assertIn("messageListRef", app_source)
        self.assertIn("shouldStickToBottomRef", app_source)
        self.assertIn("handleMessageListScroll", app_source)
        self.assertIn("scrollChatMessagesToBottom", app_source)
        self.assertIn("isChatMessageListNearBottom", app_source)
        self.assertIn("scrollTop + clientHeight", app_source)
        self.assertIn("requestAnimationFrame", app_source)
        self.assertIn("onScroll={handleMessageListScroll}", app_source)
        self.assertIn("handleChatSubmit", app_source)

    def test_chat_messages_keep_assistant_reply_after_parent_user_when_timestamps_tie(self) -> None:
        app_source = Path("apps/web/src/app/App.tsx").read_text(encoding="utf-8")

        self.assertIn("orderAgentMessagesForChat", app_source)
        self.assertIn("parent_message_id", app_source)
        self.assertIn("pendingRepliesByParentId", app_source)
        self.assertIn("flushAssistantReplies", app_source)
