import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class ToolResultEnvelopeTest(unittest.TestCase):
    def test_generic_success_envelope_preserves_business_refs_and_artifacts(self) -> None:
        from app.agent_runtime.tool_result_envelope import build_tool_result_envelope

        envelope = build_tool_result_envelope(
            capability="wechat-account.search_articles",
            status="succeeded",
            executor="agent_tool_registry",
            risk_level="low",
            result_payload={
                "tool_name": "wechat-account.search_articles",
                "ok": True,
                "summary": "抓取公众号候选文章 8 篇。",
                "result": {
                    "observations": ["命中 8 篇校招相关文章。"],
                    "artifacts": [
                        {"type": "url", "title": "腾讯 2027 校园招聘", "url": "https://mp.weixin.qq.com/s/a"}
                    ],
                    "business_refs": [{"type": "article_candidate", "id": "candidate-1"}],
                },
            },
        )

        data = envelope.to_dict()
        self.assertEqual("succeeded", data["status"])
        self.assertEqual("wechat-account.search_articles", data["capability"])
        self.assertEqual("agent_tool_registry", data["executor"])
        self.assertEqual("抓取公众号候选文章 8 篇。", data["summary"])
        self.assertEqual(["命中 8 篇校招相关文章。"], data["observations"])
        self.assertEqual(
            [{"type": "url", "title": "腾讯 2027 校园招聘", "url": "https://mp.weixin.qq.com/s/a"}],
            data["artifacts"],
        )
        self.assertEqual([{"type": "article_candidate", "id": "candidate-1"}], data["business_refs"])
        self.assertIsNone(data["error_code"])
        self.assertIsNone(data["next_action"])

    def test_generic_failure_envelope_keeps_actionable_error_contract(self) -> None:
        from app.agent_runtime.tool_result_envelope import build_tool_result_envelope

        envelope = build_tool_result_envelope(
            capability="weixin-articles-mcp.read_article",
            status="failed",
            executor="agent_tool_registry",
            risk_level="low",
            result_payload={
                "tool_name": "weixin-articles-mcp.read_article",
                "ok": False,
                "error": "公众号文章返回 403，无法直接读取正文。",
                "error_code": "WECHAT_ARTICLE_ACCESS_DENIED",
                "retryable": False,
                "next_action": "request_visible_page_read",
                "result": {"message": "需要用户在可见浏览器里打开文章后再读取。", "status_code": 403},
            },
        )

        data = envelope.to_dict()
        self.assertEqual("failed", data["status"])
        self.assertEqual("weixin-articles-mcp.read_article", data["capability"])
        self.assertEqual("WECHAT_ARTICLE_ACCESS_DENIED", data["error_code"])
        self.assertFalse(data["retryable"])
        self.assertEqual("request_visible_page_read", data["next_action"])
        self.assertTrue(data["requires_user_action"])
        self.assertIn("403", data["summary"])
        self.assertIn("需要用户在可见浏览器里打开文章后再读取。", data["observations"])

    def test_routing_result_envelope_falls_back_to_generic_tool_contract(self) -> None:
        from app.agent_runtime.routing.result_envelope import build_result_envelope

        envelope = build_result_envelope(
            capability="custom.tool",
            status="succeeded",
            risk_level="medium",
            result_payload={
                "tool_name": "custom.tool",
                "ok": True,
                "result": {"message": "工具执行完成", "business_refs": [{"type": "job_lead", "id": "lead-1"}]},
            },
        )

        self.assertIsNotNone(envelope)
        data = envelope.to_dict()
        self.assertEqual("custom.tool", data["capability"])
        self.assertEqual("medium", data["risk_level"])
        self.assertEqual("工具执行完成", data["summary"])
        self.assertEqual([{"type": "job_lead", "id": "lead-1"}], data["business_refs"])


if __name__ == "__main__":
    unittest.main()
