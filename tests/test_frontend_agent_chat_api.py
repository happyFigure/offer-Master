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
        self.assertIn("onApprovalRequired", api_source)

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
        self.assertIn('event.key === "Enter"', app_source)
        self.assertIn("event.shiftKey", app_source)
        self.assertIn("contextMetadata", app_source)
        self.assertNotIn("onCompactSession", app_source)
        self.assertNotIn("compactAgentSession", app_source)
        self.assertNotIn("手动压缩会话", app_source)
        self.assertNotIn("assistant-placeholder", app_source)

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
