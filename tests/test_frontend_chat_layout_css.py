from pathlib import Path
from unittest import TestCase


class FrontendChatLayoutCssTest(TestCase):
    def test_chat_layout_prioritizes_middle_conversation_space(self) -> None:
        css_source = Path("apps/web/src/styles/global.css").read_text(encoding="utf-8")

        app_shell = _css_block(css_source, ".app-shell")
        chat_layout = _css_block(css_source, ".chat-layout")
        message_list = _css_block(css_source, ".chat-message-list")
        assistant_bubble = _css_block(css_source, ".chat-message-assistant .chat-bubble")
        chat_panel = _css_block(css_source, ".chat-panel")

        self.assertIn("width: 100%;", app_shell)
        self.assertIn("max-width: none;", app_shell)
        self.assertIn("grid-template-columns: minmax(180px, 220px) minmax(0, 1fr) minmax(240px, 300px);", chat_layout)
        self.assertIn("min-width: 0;", chat_panel)
        self.assertIn("width: 100%;", message_list)
        self.assertIn("justify-self: stretch;", message_list)
        self.assertIn("width: min(100%, 1280px);", assistant_bubble)
        self.assertIn("@media (max-width: 1500px)", css_source)
        self.assertIn("grid-column: 1 / -1;", css_source)

    def test_chat_messages_follow_deepseek_style_reading_flow(self) -> None:
        css_source = Path("apps/web/src/styles/global.css").read_text(encoding="utf-8")

        message_list = _css_block(css_source, ".chat-message-list")
        assistant_bubble = _css_block(css_source, ".chat-message-assistant .chat-bubble")
        user_bubble = _css_block(css_source, ".chat-message-user .chat-bubble")
        user_meta = _css_block(css_source, ".chat-message-user .chat-meta")
        avatar = _css_block(css_source, ".chat-message .chat-avatar")

        self.assertIn("border: 0;", message_list)
        self.assertIn("background: transparent;", message_list)
        self.assertIn("background: transparent;", assistant_bubble)
        self.assertIn("border: 0;", assistant_bubble)
        self.assertIn("border-radius: 999px;", user_bubble)
        self.assertIn("width: max-content;", user_bubble)
        self.assertIn("display: none;", user_meta)
        self.assertIn("display: none;", avatar)

    def test_chat_tool_approval_card_has_explicit_actions(self) -> None:
        css_source = Path("apps/web/src/styles/global.css").read_text(encoding="utf-8")

        approval_card = _css_block(css_source, ".tool-approval-card")
        approval_actions = _css_block(css_source, ".tool-approval-actions")

        self.assertIn("border", approval_card)
        self.assertIn("grid", approval_card)
        self.assertIn("display: flex;", approval_actions)

    def test_chat_runtime_timeline_is_compact_and_status_driven(self) -> None:
        css_source = Path("apps/web/src/styles/global.css").read_text(encoding="utf-8")

        timeline = _css_block(css_source, ".chat-runtime-timeline")
        heading = _css_block(css_source, ".chat-runtime-heading")
        event_item = _css_block(css_source, ".runtime-event-item")

        self.assertIn("display: grid;", timeline)
        self.assertIn("max-height", timeline)
        self.assertIn("min-height: 420px;", timeline)
        self.assertIn("grid-template-rows: minmax(0, 1fr) auto;", css_source)
        self.assertIn("max-height: calc(100dvh - 162px);", css_source)
        self.assertIn("display: flex;", heading)
        self.assertIn("grid-template-columns: auto minmax(0, 1fr);", event_item)
        self.assertIn(".runtime-event-running", css_source)
        self.assertIn(".runtime-event-success", css_source)
        self.assertIn(".runtime-event-warning", css_source)
        self.assertIn(".runtime-event-danger", css_source)


def _css_block(css_source: str, selector: str) -> str:
    marker = f"{selector} {{"
    if marker not in css_source:
        return ""
    start = css_source.index(marker)
    end = css_source.index("}\n", start)
    return css_source[start:end]
