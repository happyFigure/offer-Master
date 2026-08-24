import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class ReflectionEvaluatorTest(unittest.TestCase):
    def test_web_search_reflection_retries_when_result_misses_company_and_campus(self) -> None:
        from app.agent_runtime.reflection.evaluator import ReflectionEvaluator
        from app.agent_runtime.reflection.schemas import ReflectionNextAction, ReflectionQuality

        decision = ReflectionEvaluator().evaluate_web_search_result(
            tool_input={"query": "中科曙光 校园招聘 官网", "max_results": 5},
            result_payload={
                "tool_name": "external.web_search",
                "ok": True,
                "result": {
                    "answer": "中（汉语汉字）_百度百科",
                    "sources": [{"title": "中（汉语汉字）_百度百科", "url": "https://baike.baidu.com/item/中"}],
                },
            },
            expected_company_names=["中科曙光"],
        )

        self.assertEqual(ReflectionQuality.BAD, decision.quality)
        self.assertEqual(ReflectionNextAction.RETRY, decision.next_action)
        self.assertLess(decision.confidence, 0.5)
        self.assertIn("中科曙光", decision.suggested_input_patch["query"])
        self.assertIn("校园招聘", decision.suggested_input_patch["query"])
        self.assertIn("not match", decision.reason)

    def test_web_search_reflection_continues_when_official_campus_result_is_found(self) -> None:
        from app.agent_runtime.reflection.evaluator import ReflectionEvaluator
        from app.agent_runtime.reflection.schemas import ReflectionNextAction, ReflectionQuality

        decision = ReflectionEvaluator().evaluate_web_search_result(
            tool_input={"query": "腾讯 校园招聘 官网", "max_results": 5},
            result_payload={
                "tool_name": "external.web_search",
                "ok": True,
                "result": {
                    "answer": "腾讯校招官网：https://join.qq.com/",
                    "sources": [{"title": "腾讯校招官网", "url": "https://join.qq.com/"}],
                },
            },
            expected_company_names=["腾讯"],
        )

        self.assertEqual(ReflectionQuality.GOOD, decision.quality)
        self.assertEqual(ReflectionNextAction.CONTINUE, decision.next_action)
        self.assertGreaterEqual(decision.confidence, 0.8)
        self.assertEqual({}, decision.suggested_input_patch)
        self.assertEqual("structured", decision.metadata["mode"])

    def test_public_sports_web_search_retry_does_not_use_campus_query(self) -> None:
        from app.agent_runtime.reflection.evaluator import ReflectionEvaluator
        from app.agent_runtime.reflection.schemas import ReflectionNextAction, ReflectionQuality

        decision = ReflectionEvaluator().evaluate_web_search_result(
            tool_input={
                "query": "Cristiano Ronaldo C罗 Al Nassr Portugal football fixtures match schedule this week week of 2026-08-24",
                "max_results": 5,
            },
            result_payload={
                "tool_name": "external.web_search",
                "ok": True,
                "result": {
                    "answer": "检索结果均为UTF-8编码转换类工具网站，与足球赛程无关。",
                    "sources": [{"title": "UTF-8 编码转换", "url": "https://example.com/utf8"}],
                },
            },
            expected_company_names=[],
        )

        self.assertEqual(ReflectionQuality.BAD, decision.quality)
        self.assertEqual(ReflectionNextAction.RETRY, decision.next_action)
        self.assertIn("Cristiano Ronaldo", decision.suggested_input_patch["query"])
        self.assertIn("Al Nassr", decision.suggested_input_patch["query"])
        self.assertIn("fixtures", decision.suggested_input_patch["query"])
        self.assertNotIn("校园招聘", decision.suggested_input_patch["query"])

    def test_public_sports_web_search_rejects_fixture_result_without_target_player_or_team(self) -> None:
        from app.agent_runtime.reflection.evaluator import ReflectionEvaluator
        from app.agent_runtime.reflection.schemas import ReflectionNextAction, ReflectionQuality

        decision = ReflectionEvaluator().evaluate_web_search_result(
            tool_input={
                "query": "Cristiano Ronaldo C罗 Al Nassr Portugal football fixtures match schedule this week week of 2026-08-24",
                "max_results": 5,
            },
            result_payload={
                "tool_name": "external.web_search",
                "ok": True,
                "result": {
                    "answer": "2026世界杯赛程表，涵盖小组赛、淘汰赛及决赛全部比赛时间。",
                    "sources": [{"title": "2026世界杯赛程", "url": "https://example.com/worldcup"}],
                },
            },
            expected_company_names=[],
        )

        self.assertEqual(ReflectionQuality.BAD, decision.quality)
        self.assertEqual(ReflectionNextAction.RETRY, decision.next_action)
        self.assertIn("Cristiano Ronaldo", decision.suggested_input_patch["query"])
        self.assertIn("Al Nassr", decision.suggested_input_patch["query"])

    def test_public_sports_web_search_retry_preserves_last_match_direction(self) -> None:
        from app.agent_runtime.reflection.evaluator import ReflectionEvaluator
        from app.agent_runtime.reflection.schemas import ReflectionNextAction, ReflectionQuality

        decision = ReflectionEvaluator().evaluate_web_search_result(
            tool_input={
                "query": "Cristiano Ronaldo C罗 Al Nassr Portugal last match result date",
                "max_results": 5,
            },
            result_payload={
                "tool_name": "external.web_search",
                "ok": True,
                "result": {
                    "answer": "检索结果均为UTF-8编码转换类工具网站，与足球比赛无关。",
                    "sources": [{"title": "UTF-8 编码转换", "url": "https://example.com/utf8"}],
                },
            },
            expected_company_names=[],
        )

        self.assertEqual(ReflectionQuality.BAD, decision.quality)
        self.assertEqual(ReflectionNextAction.RETRY, decision.next_action)
        self.assertIn("Cristiano Ronaldo", decision.suggested_input_patch["query"])
        self.assertIn("last match", decision.suggested_input_patch["query"])
        self.assertIn("result date", decision.suggested_input_patch["query"])
        self.assertNotIn("next match", decision.suggested_input_patch["query"])

    def test_public_sports_web_search_rejects_answer_without_traceable_source(self) -> None:
        from app.agent_runtime.reflection.evaluator import ReflectionEvaluator
        from app.agent_runtime.reflection.schemas import ReflectionNextAction, ReflectionQuality

        decision = ReflectionEvaluator().evaluate_web_search_result(
            tool_input={
                "query": "Cristiano Ronaldo C罗 Al Nassr Portugal last match result date ESPN Flashscore SofaScore",
                "max_results": 5,
            },
            result_payload={
                "tool_name": "external.web_search",
                "ok": True,
                "result": {
                    "answer": "Cristiano Ronaldo's most recent match was an Al Nassr match according to ESPN and Flashscore.",
                    "sources": [],
                    "artifacts": [],
                },
            },
            expected_company_names=[],
        )

        self.assertEqual(ReflectionQuality.BAD, decision.quality)
        self.assertEqual(ReflectionNextAction.RETRY, decision.next_action)
        self.assertIn("source", decision.reason)
        self.assertIn("last match", decision.suggested_input_patch["query"])


if __name__ == "__main__":
    unittest.main()
