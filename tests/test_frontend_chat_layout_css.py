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
        self.assertIn("grid-template-columns: minmax(180px, 220px) minmax(0, 1fr) minmax(440px, 560px);", chat_layout)
        self.assertIn("min-width: 0;", chat_panel)
        self.assertIn("width: 100%;", message_list)
        self.assertIn("justify-self: stretch;", message_list)
        self.assertIn("width: min(100%, 1280px);", assistant_bubble)
        self.assertIn("@media (max-width: 1360px)", css_source)
        self.assertIn("grid-column: 1 / -1;", css_source)

    def test_chat_runtime_timeline_avoids_horizontal_scroll(self) -> None:
        css_source = Path("apps/web/src/styles/global.css").read_text(encoding="utf-8")

        timeline = _css_block(css_source, ".chat-runtime-timeline")
        copy = _css_block(css_source, ".runtime-event-copy")
        detail = _css_block(css_source, ".runtime-event-detail-grid span")
        evidence = _css_block(css_source, ".runtime-event-evidence a,\n.runtime-event-evidence span")

        self.assertIn("overflow-x: hidden;", timeline)
        self.assertIn("overflow-wrap: anywhere;", copy)
        self.assertIn("overflow-wrap: anywhere;", detail)
        self.assertIn("max-width: 100%;", evidence)

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
        self.assertIn("border-radius: 18px;", user_bubble)
        self.assertIn("width: fit-content;", user_bubble)
        self.assertIn("overflow-wrap: anywhere;", user_bubble)
        self.assertNotIn("border-radius: 999px;", user_bubble)
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
        self.assertIn("min-height: 300px;", timeline)
        self.assertIn("grid-template-rows: auto minmax(0, 1fr) auto;", css_source)
        self.assertIn("max-height: calc(100dvh - 96px);", css_source)
        self.assertIn("display: flex;", heading)
        self.assertIn("grid-template-columns: 34px minmax(0, 1fr);", event_item)
        self.assertIn(".runtime-event-running", css_source)
        self.assertIn(".runtime-event-success", css_source)
        self.assertIn(".runtime-event-warning", css_source)
        self.assertIn(".runtime-event-danger", css_source)

    def test_chat_runtime_timeline_uses_actor_cards_without_inner_horizontal_scroll(self) -> None:
        css_source = Path("apps/web/src/styles/global.css").read_text(encoding="utf-8")

        side_panel = _css_block(css_source, ".chat-side-panel", contains="display: grid;")
        event_item = _css_block(css_source, ".runtime-event-item")
        event_card = _css_block(css_source, ".runtime-event-card")
        payload_preview = _css_block(css_source, ".runtime-event-payload-preview")
        actor_badge = _css_block(css_source, ".runtime-actor-badge")

        self.assertIn("minmax(440px, 560px)", css_source)
        self.assertIn("overflow: hidden;", side_panel)
        self.assertIn("grid-template-columns: 34px minmax(0, 1fr);", event_item)
        self.assertIn("min-width: 0;", event_card)
        self.assertIn("overflow-x: hidden;", payload_preview)
        self.assertIn("white-space: normal;", payload_preview)
        self.assertIn("runtime-actor-badge-main", css_source)
        self.assertIn("runtime-actor-badge-subagent", css_source)
        self.assertIn("runtime-actor-badge-tool", css_source)

    def test_chat_side_panel_gives_more_height_to_task_plan(self) -> None:
        css_source = Path("apps/web/src/styles/global.css").read_text(encoding="utf-8")

        task_plan = _css_block(css_source, ".chat-task-plan-panel")
        timeline = _css_block(css_source, ".chat-runtime-timeline")

        self.assertIn("max-height: clamp(360px, 44dvh, 560px);", task_plan)
        self.assertIn("min-height: 300px;", timeline)
        self.assertIn("max-height: calc(100dvh - 210px);", timeline)

    def test_chat_runtime_timeline_animates_new_cards_without_layout_shift(self) -> None:
        css_source = Path("apps/web/src/styles/global.css").read_text(encoding="utf-8")
        event_item = _css_block(css_source, ".runtime-event-item")

        self.assertIn("animation: runtime-event-enter", event_item)
        self.assertIn("transform", event_item)
        self.assertIn("@keyframes runtime-event-enter", css_source)
        self.assertIn("translateY(8px)", css_source)
        self.assertIn("scale(0.985)", css_source)
        self.assertIn("prefers-reduced-motion: reduce", css_source)
        self.assertIn(".runtime-event-item", css_source)
        self.assertIn("animation: none !important;", css_source)


def _css_block(css_source: str, selector: str, *, contains: str | None = None) -> str:
    marker = f"{selector} {{"
    if marker not in css_source:
        return ""
    start = 0
    while True:
        try:
            start = css_source.index(marker, start)
        except ValueError:
            return ""
        end = css_source.index("}\n", start)
        block = css_source[start:end]
        if contains is None or contains in block:
            return block
        start = end + 1
