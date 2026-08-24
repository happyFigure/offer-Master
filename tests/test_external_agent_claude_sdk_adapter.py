import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


def sample_envelope(task_id="task-claude-1"):
    from app.agent_runtime.external_tasks.schemas import (
        ExternalTaskCandidateProfileRef,
        ExternalTaskJobContext,
        FindApplyEntryTaskEnvelope,
    )

    return FindApplyEntryTaskEnvelope(
        task_id=task_id,
        trace_id="trace-claude-1",
        job=ExternalTaskJobContext(
            job_id="lead-claude-1",
            company_name="Tencent",
            title="Backend Engineer Intern",
            source_url="https://careers.tencent.com/job/1",
            apply_url_candidate="https://careers.tencent.com/apply/1",
            jd_summary="Campus backend role requiring Java.",
        ),
        candidate_profile_ref=ExternalTaskCandidateProfileRef(
            profile_id="default",
            resume_version_id="resume-v3",
        ),
    )


class ClaudeSdkHttpExecutorAdapterTest(unittest.TestCase):
    def test_adapter_posts_openai_compatible_request_and_parses_json_result(self):
        from app.agent_runtime.external_tasks.executors import (
            ClaudeSdkHttpExecutorAdapter,
            ClaudeSdkHttpExecutorConfig,
        )
        from app.agent_runtime.external_tasks.schemas import ApplyEntryDiscoveryStatus

        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            captured["payload"] = json.loads(request.content.decode("utf-8"))
            content = json.dumps(
                {
                    "task_id": "task-claude-1",
                    "status": "found_opened",
                    "confidence": 0.87,
                    "company_name": "Tencent",
                    "job_title": "Backend Engineer Intern",
                    "source_url": "https://careers.tencent.com/job/1",
                    "apply_url": "https://careers.tencent.com/apply/1",
                    "final_browser_url": "https://careers.tencent.com/apply/1",
                    "platform": "tencent_careers",
                    "button_text": "立即投递",
                    "requires_login": False,
                    "candidate_urls": ["https://careers.tencent.com/apply/1"],
                    "evidence_artifacts": [
                        {
                            "artifact_type": "web_search_result",
                            "path_or_uri": "https://careers.tencent.com/apply/1",
                            "mime_type": "text/html",
                            "metadata": {"reason": "official careers apply URL"},
                        }
                    ],
                    "notes": "Found via official careers page.",
                    "next_action": "wait_user_review",
                },
                ensure_ascii=False,
            )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": f"```json\n{content}\n```"}}]},
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        adapter = ClaudeSdkHttpExecutorAdapter(
            config=ClaudeSdkHttpExecutorConfig(
                base_url="http://127.0.0.1:18008",
                model="MiniMax-M2.7",
                api_key="tdl-key",
                timeout_seconds=5.0,
            ),
            client=client,
        )

        result = adapter.execute_find_apply_entry(sample_envelope())

        self.assertEqual(ApplyEntryDiscoveryStatus.FOUND_OPENED, result.status)
        self.assertEqual("https://careers.tencent.com/apply/1", result.apply_url)
        self.assertEqual("web_search_result", result.evidence_artifacts[0].artifact_type)
        self.assertEqual("http://127.0.0.1:18008/v1/chat/completions", captured["url"])
        self.assertEqual("tdl-key", captured["headers"].get("x-api-key"))
        payload = captured["payload"]
        self.assertFalse(payload["stream"])
        self.assertEqual("MiniMax-M2.7", payload["model"])
        self.assertEqual("task-claude-1", payload["session_id"])
        self.assertEqual("task-claude-1", payload["user"])
        rendered_messages = json.dumps(payload["messages"], ensure_ascii=False)
        self.assertIn("offer_master.find_apply_entry_task.v1", rendered_messages)
        self.assertIn("submit_application", rendered_messages)
        self.assertIn("Return only one JSON object", rendered_messages)

    def test_adapter_raises_executor_error_for_malformed_assistant_json(self):
        from app.agent_runtime.external_tasks.executors import (
            ClaudeSdkHttpExecutorAdapter,
            ClaudeSdkHttpExecutorConfig,
            ExternalExecutorError,
        )

        client = httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    json={"choices": [{"message": {"role": "assistant", "content": "not json"}}]},
                )
            )
        )
        adapter = ClaudeSdkHttpExecutorAdapter(
            config=ClaudeSdkHttpExecutorConfig(base_url="http://127.0.0.1:18008"),
            client=client,
        )

        with self.assertRaisesRegex(ExternalExecutorError, "valid JSON"):
            adapter.execute_find_apply_entry(sample_envelope("task-bad-json"))

    def test_adapter_posts_external_web_search_request_and_returns_assistant_content(self):
        from app.agent_runtime.external_tasks.executors import (
            ClaudeSdkHttpExecutorAdapter,
            ClaudeSdkHttpExecutorConfig,
        )

        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            captured["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "梅西最近一场比赛是迈阿密国际的 MLS 比赛。来源：https://example.com/messi",
                            }
                        }
                    ]
                },
            )

        adapter = ClaudeSdkHttpExecutorAdapter(
            config=ClaudeSdkHttpExecutorConfig(
                base_url="http://127.0.0.1:18008",
                model="qwen-plus",
                api_key="tdl-key",
                timeout_seconds=5.0,
                provider_base_url="https://dashscope.aliyuncs.com/apps/anthropic",
                provider_api_key="sk-provider",
            ),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        result = adapter.execute_web_search("你给我搜索一下梅西最近的比赛", max_results=5)

        self.assertEqual("http://127.0.0.1:18008/v1/chat/completions", captured["url"])
        self.assertEqual("tdl-key", captured["headers"].get("x-api-key"))
        payload = captured["payload"]
        self.assertFalse(payload["stream"])
        self.assertEqual("qwen-plus", payload["model"])
        self.assertTrue(str(payload["session_id"]).startswith("external-web-search-"))
        self.assertNotEqual("external-web-search-3300828034530744390", payload["session_id"])
        llm_config = payload["metadata"]["agentconfig"]["runtime_config"]["llm"]
        self.assertEqual("https://dashscope.aliyuncs.com/apps/anthropic", llm_config["base_url"])
        self.assertEqual("sk-provider", llm_config["api_key"])
        self.assertEqual("2023-06-01", llm_config["anthropic_version"])
        rendered_messages = json.dumps(payload["messages"], ensure_ascii=False)
        self.assertIn("WebSearch", rendered_messages)
        self.assertIn("source URLs", rendered_messages)
        self.assertIn("Source names alone are not enough", rendered_messages)
        self.assertIn("你给我搜索一下梅西最近的比赛", rendered_messages)
        self.assertEqual("claude-sdk-agent", result["executor_name"])
        self.assertEqual("你给我搜索一下梅西最近的比赛", result["query"])
        self.assertIn("梅西最近一场比赛", result["answer"])
        self.assertEqual(["梅西最近一场比赛是迈阿密国际的 MLS 比赛。来源：https://example.com/messi"], result["observations"])
        self.assertEqual(
            [{"type": "url", "title": "source", "url": "https://example.com/messi"}],
            result["artifacts"],
        )

    def test_adapter_falls_back_to_http_search_when_claude_returns_empty_web_search_content(self):
        from app.agent_runtime.external_tasks.executors import (
            ClaudeSdkHttpExecutorAdapter,
            ClaudeSdkHttpExecutorConfig,
        )

        fallback_calls = []

        def fallback(query, *, max_results):
            fallback_calls.append((query, max_results))
            return {
                "executor_name": "http-web-search-fallback",
                "query": query,
                "answer": "联网搜索结果：梅西最近比赛来自兜底搜索。",
                "sources": ["https://example.com/messi-latest-match"],
            }

        client = httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    json={"choices": [{"message": {"role": "assistant", "content": ""}}]},
                )
            )
        )
        adapter = ClaudeSdkHttpExecutorAdapter(
            config=ClaudeSdkHttpExecutorConfig(base_url="http://127.0.0.1:18008"),
            client=client,
            web_search_fallback=fallback,
        )

        result = adapter.execute_web_search("你给我搜索一下梅西最近的比赛", max_results=3)

        self.assertEqual([("你给我搜索一下梅西最近的比赛", 3)], fallback_calls)
        self.assertEqual("http-web-search-fallback", result["executor_name"])
        self.assertIn("兜底搜索", result["answer"])

    def test_adapter_falls_back_when_claude_returns_tool_call_marker_without_final_answer(self):
        from app.agent_runtime.external_tasks.executors import (
            ClaudeSdkHttpExecutorAdapter,
            ClaudeSdkHttpExecutorConfig,
        )

        fallback_calls = []

        def fallback(query, *, max_results):
            fallback_calls.append((query, max_results))
            return {
                "executor_name": "http-web-search-fallback",
                "query": query,
                "answer": "联网搜索结果：兜底搜索。",
                "sources": ["https://example.com/fallback"],
            }

        client = httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": '(tool_call)\n{"name":"WebSearch","arguments":{"query":"Cristiano Ronaldo latest match"}}\n(tool_call)',
                                }
                            }
                        ]
                    },
                )
            )
        )
        adapter = ClaudeSdkHttpExecutorAdapter(
            config=ClaudeSdkHttpExecutorConfig(base_url="http://127.0.0.1:18008"),
            client=client,
            web_search_fallback=fallback,
        )

        result = adapter.execute_web_search("Cristiano Ronaldo last match", max_results=3)

        self.assertEqual([("Cristiano Ronaldo last match", 3)], fallback_calls)
        self.assertEqual("http-web-search-fallback", result["executor_name"])

    def test_bing_search_parser_extracts_result_cards(self):
        from app.agent_runtime.external_tasks.executors import _parse_bing_results

        html = """
        <ol id="b_results">
          <li class="b_algo">
            <h2><a href="https://example.com/messi">梅西最近比赛结果</a></h2>
            <div class="b_caption"><p>迈阿密国际最近一场比赛摘要。</p></div>
          </li>
        </ol>
        """

        results = _parse_bing_results(html, limit=3)

        self.assertEqual(
            [{"title": "梅西最近比赛结果", "url": "https://example.com/messi", "snippet": "迈阿密国际最近一场比赛摘要。"}],
            results,
        )

    def test_http_web_search_uses_exact_query_without_cleaning(self):
        from app.agent_runtime.external_tasks.executors import _run_http_web_search

        calls = []

        def fake_bing(query: str, *, max_results: int = 5):
            calls.append((query, max_results))
            return {
                "executor_name": "fake-search",
                "query": query,
                "answer": "联网搜索结果：ok",
                "results": [],
                "sources": [],
            }

        with patch("app.agent_runtime.external_tasks.executors._run_bing_web_search", side_effect=fake_bing):
            result = _run_http_web_search("你给我搜索一下中科曙光的校园招聘信息", max_results=3)

        self.assertEqual([("你给我搜索一下中科曙光的校园招聘信息", 3)], calls)
        self.assertEqual("你给我搜索一下中科曙光的校园招聘信息", result["query"])

    def test_http_web_search_payload_includes_structured_observations_and_artifacts(self):
        from app.agent_runtime.external_tasks.executors import _web_search_result_payload

        result = _web_search_result_payload(
            "中科曙光 校园招聘",
            [
                {
                    "title": "中科曙光校园招聘",
                    "url": "https://jobs.example.com/sugon",
                    "snippet": "官方校招入口。",
                }
            ],
            executor_name="http-web-search-fallback",
        )

        self.assertIn("中科曙光校园招聘", result["observations"][0])
        self.assertEqual(
            [{"type": "url", "title": "中科曙光校园招聘", "url": "https://jobs.example.com/sugon"}],
            result["artifacts"],
        )


if __name__ == "__main__":
    unittest.main()
