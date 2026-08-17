import sys
import unittest
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class LLMChatClientTest(unittest.TestCase):
    def test_complete_posts_openai_compatible_payload_and_returns_assistant_text(self):
        from app.infrastructure.llm.chat_client import LLMChatClient
        from app.infrastructure.llm.client import LLMRuntimeConfig

        captured = {}

        def handler(request):
            captured["authorization"] = request.headers.get("authorization")
            captured["url"] = str(request.url)
            captured["payload"] = request.read().decode("utf-8")
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": "可以，我会先整理你的秋招目标。"}}
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 8},
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        config = LLMRuntimeConfig(
            provider="bailian",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-test",
            model="qwen-plus",
            timeout_seconds=5.0,
            max_retries=0,
        )

        result = LLMChatClient(config=config, client=client).complete(
            messages=[{"role": "user", "content": "帮我规划 Java 秋招"}],
        )

        self.assertEqual("Bearer sk-test", captured["authorization"])
        self.assertEqual("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", captured["url"])
        self.assertIn('"model":"qwen-plus"', captured["payload"])
        self.assertIn('"role":"user"', captured["payload"])
        self.assertEqual("可以，我会先整理你的秋招目标。", result.content)
        self.assertEqual({"prompt_tokens": 12, "completion_tokens": 8}, result.usage)

    def test_stream_complete_parses_openai_compatible_sse_deltas(self):
        from app.infrastructure.llm.chat_client import LLMChatClient
        from app.infrastructure.llm.client import LLMRuntimeConfig

        captured = {}

        def handler(request):
            captured["payload"] = request.read().decode("utf-8")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    'data: {"choices":[{"delta":{"content":"你"}}]}\n\n'
                    'data: {"choices":[{"delta":{"content":"好"}}]}\n\n'
                    "data: [DONE]\n\n"
                ),
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        config = LLMRuntimeConfig(
            provider="bailian",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-test",
            model="qwen-plus",
            timeout_seconds=5.0,
            max_retries=0,
        )

        chunks = list(LLMChatClient(config=config, client=client).stream_complete(messages=[{"role": "user", "content": "hi"}]))

        self.assertIn('"stream":true', captured["payload"])
        self.assertEqual(["你", "好"], chunks)


if __name__ == "__main__":
    unittest.main()
