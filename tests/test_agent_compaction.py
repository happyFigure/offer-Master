import sys
import unittest
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


@dataclass(frozen=True)
class MessageStub:
    id: str
    role: str
    content_text: str | None = None
    visible_content_text: str | None = None
    runtime_content_text: str | None = None
    token_estimate: int | None = None
    parent_message_id: str | None = None


class AgentCompactionTest(unittest.TestCase):
    def test_estimate_tokens_handles_empty_chinese_and_longer_text(self):
        from app.agent_runtime.memory.token_budget import estimate_tokens

        self.assertEqual(0, estimate_tokens(None))
        self.assertEqual(0, estimate_tokens(""))
        short = estimate_tokens("帮我找 Java 后端岗位")
        long = estimate_tokens("帮我找 Java 后端岗位" * 30)

        self.assertGreater(short, 0)
        self.assertGreater(long, short)

    def test_should_compact_uses_reserved_output_budget(self):
        from app.agent_runtime.memory.token_budget import should_compact

        self.assertFalse(should_compact(context_tokens=47000, context_window=64000, reserve_tokens=16384))
        self.assertTrue(should_compact(context_tokens=48000, context_window=64000, reserve_tokens=16384))

    def test_find_cut_point_keeps_recent_messages_by_token_budget_not_fixed_turns(self):
        from app.agent_runtime.memory.compaction import find_cut_point

        messages = [
            MessageStub(id=f"m{i:02d}", role="user", content_text=f"message {i}", token_estimate=1000)
            for i in range(30)
        ]

        cut = find_cut_point(messages, keep_recent_tokens=20000)

        self.assertEqual(10, len(cut.messages_to_summarize))
        self.assertEqual(20, len(cut.recent_messages_to_keep))
        self.assertEqual("m10", cut.first_kept_message_id)
        self.assertEqual("m29", cut.recent_messages_to_keep[-1].id)

    def test_find_cut_point_does_not_split_adjacent_tool_call_and_result(self):
        from app.agent_runtime.memory.compaction import find_cut_point

        messages = [
            MessageStub(id="m1", role="user", content_text="start", token_estimate=100),
            MessageStub(id="tool-call-1", role="tool_call", content_text="fetch url", token_estimate=100),
            MessageStub(
                id="tool-result-1",
                role="tool_result",
                content_text="saved raw payload",
                token_estimate=100,
                parent_message_id="tool-call-1",
            ),
            MessageStub(id="m4", role="assistant", content_text="next", token_estimate=100),
        ]

        cut = find_cut_point(messages, keep_recent_tokens=250)

        self.assertEqual(["m1"], [message.id for message in cut.messages_to_summarize])
        self.assertEqual(
            ["tool-call-1", "tool-result-1", "m4"],
            [message.id for message in cut.recent_messages_to_keep],
        )
        self.assertEqual("tool-call-1", cut.first_kept_message_id)

    def test_prepare_compaction_returns_summary_inputs_and_previous_summary(self):
        from app.agent_runtime.memory.compaction import CompactionConfig, prepare_compaction

        messages = [
            MessageStub(id="old-1", role="user", content_text="old context", token_estimate=9000),
            MessageStub(id="old-2", role="assistant", content_text="old answer", token_estimate=9000),
            MessageStub(id="recent-1", role="user", content_text="recent context", token_estimate=1000),
            MessageStub(id="recent-2", role="assistant", content_text="recent answer", token_estimate=1000),
        ]

        plan = prepare_compaction(
            messages,
            latest_summary="Progress: 已经完成信息源同步。",
            config=CompactionConfig(context_window=24000, reserve_tokens=4096, keep_recent_tokens=2500),
        )

        self.assertTrue(plan.should_compact)
        self.assertEqual(["old-1", "old-2"], [message.id for message in plan.messages_to_summarize])
        self.assertEqual(["recent-1", "recent-2"], [message.id for message in plan.recent_messages_to_keep])
        self.assertEqual("recent-1", plan.first_kept_message_id)
        self.assertEqual("Progress: 已经完成信息源同步。", plan.previous_summary)

    def test_build_summary_prompt_uses_openclaw_style_sections(self):
        from app.agent_runtime.memory.compaction import CompactionConfig, build_summary_prompt, prepare_compaction

        messages = [
            MessageStub(id="old-1", role="user", content_text="用户要找 Java 后端秋招", token_estimate=8000),
            MessageStub(id="old-2", role="assistant", content_text="已经记录偏好", token_estimate=8000),
            MessageStub(id="recent-1", role="user", content_text="下一步是什么", token_estimate=1000),
        ]
        plan = prepare_compaction(
            messages,
            latest_summary=None,
            config=CompactionConfig(context_window=20000, reserve_tokens=4096, keep_recent_tokens=1500),
        )

        prompt = build_summary_prompt(plan)

        for section in [
            "Goal",
            "Constraints & Preferences",
            "Progress",
            "Key Decisions",
            "Next Steps",
            "Critical Context",
            "Retrieval Hints",
        ]:
            self.assertIn(section, prompt)
        self.assertIn("用户要找 Java 后端秋招", prompt)
        self.assertIn("old-1", prompt)


if __name__ == "__main__":
    unittest.main()
