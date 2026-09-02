import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class AgentOutputSanitizerTest(unittest.TestCase):
    def test_removes_tool_call_line_but_keeps_user_facing_answer(self) -> None:
        from app.agent_runtime.output_sanitizer import sanitize_agent_final_answer

        result = sanitize_agent_final_answer(
            'Tool call: external.web_search{"query":"Canonical Ltd. 主要业务"}\n\n'
            "Canonical Ltd. 主要做 Ubuntu、企业 Linux、云基础设施和安全支持。"
        )

        self.assertEqual("Canonical Ltd. 主要做 Ubuntu、企业 Linux、云基础设施和安全支持。", result.content)
        self.assertTrue(result.removed_internal_protocol)
        self.assertFalse(result.needs_regeneration)

    def test_removes_internal_protocol_blocks(self) -> None:
        from app.agent_runtime.output_sanitizer import sanitize_agent_final_answer

        result = sanitize_agent_final_answer(
            "我查到了结果。\n"
            '[toolCall]{"name":"external.web_search","arguments":{"query":"腾讯 校招"}}[/toolCall]\n'
            '[run_result]{"status":"ok"}[/run_result]\n'
            "腾讯校招入口可以从官网招聘页继续确认。"
        )

        self.assertEqual("我查到了结果。\n腾讯校招入口可以从官网招聘页继续确认。", result.content)
        self.assertTrue(result.removed_internal_protocol)
        self.assertFalse(result.needs_regeneration)

    def test_protocol_only_answer_requires_regeneration(self) -> None:
        from app.agent_runtime.output_sanitizer import sanitize_agent_final_answer

        result = sanitize_agent_final_answer('Tool call: external.web_search{"query":"Canonical Ltd."}')

        self.assertEqual("", result.content)
        self.assertTrue(result.removed_internal_protocol)
        self.assertTrue(result.needs_regeneration)

    def test_protocol_only_answer_with_assistant_label_requires_regeneration(self) -> None:
        from app.agent_runtime.output_sanitizer import sanitize_agent_final_answer

        result = sanitize_agent_final_answer(
            '**OfferMaster AI**\nTool call: external.web_search{"query":"C罗 本周 比赛日程"}'
        )

        self.assertEqual("", result.content)
        self.assertTrue(result.removed_internal_protocol)
        self.assertTrue(result.needs_regeneration)

    def test_keeps_normal_user_facing_tool_words(self) -> None:
        from app.agent_runtime.output_sanitizer import sanitize_agent_final_answer

        result = sanitize_agent_final_answer("我会先用公开信息核对公司业务，再给你总结。")

        self.assertEqual("我会先用公开信息核对公司业务，再给你总结。", result.content)
        self.assertFalse(result.removed_internal_protocol)
        self.assertFalse(result.needs_regeneration)


if __name__ == "__main__":
    unittest.main()
