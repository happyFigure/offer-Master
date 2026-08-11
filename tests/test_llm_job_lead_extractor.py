import sys
import unittest
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class LLMJobLeadExtractorTest(unittest.TestCase):
    def test_extract_parses_openai_compatible_json_items(self):
        from app.infrastructure.llm.client import LLMRuntimeConfig
        from app.infrastructure.llm.job_lead_extractor import LLMJobLeadExtractor

        captured = {}

        def handler(request):
            captured["authorization"] = request.headers.get("authorization")
            captured["url"] = str(request.url)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"items":[{"company_name":"Ant Group",'
                                    '"title":"Agent Application Engineer",'
                                    '"city":"Hangzhou","job_direction":"agent_ai",'
                                    '"graduation_year":"2027","skills":["Python","LLM"],'
                                    '"confidence_score":92}]}'
                                )
                            }
                        }
                    ]
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
        extractor = LLMJobLeadExtractor(config=config, client=client)

        leads = extractor.extract(
            "Ant Group 2027 autumn recruiting Agent application engineer",
            {"source_url": "https://example.com/note", "trust_level": "medium"},
        )

        self.assertEqual("Bearer sk-test", captured["authorization"])
        self.assertEqual(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            captured["url"],
        )
        self.assertEqual(1, len(leads))
        self.assertEqual("Ant Group", leads[0].company_name)
        self.assertEqual("Agent Application Engineer", leads[0].title)
        self.assertEqual(["Python", "LLM"], leads[0].skills)


if __name__ == "__main__":
    unittest.main()
