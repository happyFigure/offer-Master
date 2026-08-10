import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class LLMConfigurationTest(unittest.TestCase):
    def test_settings_loads_bailian_openai_compatible_defaults(self):
        from app.core.config import Settings

        with patch.dict(os.environ, {"JOBPILOT_LLM_API_KEY": "sk-test"}, clear=False):
            settings = Settings(_env_file=None)

        self.assertEqual("bailian", settings.llm_provider)
        self.assertEqual(
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            settings.llm_base_url,
        )
        self.assertEqual("qwen-plus", settings.llm_model)
        self.assertEqual("sk-test", settings.llm_api_key.get_secret_value())
        self.assertEqual("disabled", settings.embedding_provider)
        self.assertEqual("deferred", settings.vector_store_provider)

    def test_llm_runtime_config_redacts_secret(self):
        from app.core.config import Settings
        from app.infrastructure.llm.client import build_llm_runtime_config

        settings = Settings(
            _env_file=None,
            JOBPILOT_LLM_API_KEY="sk-test-secret",
        )

        config = build_llm_runtime_config(settings)

        self.assertEqual("bailian", config.provider)
        self.assertEqual("qwen-plus", config.model)
        self.assertEqual("sk-test-secret", config.api_key)
        self.assertNotIn("sk-test-secret", config.safe_summary().values())
        self.assertEqual("**********", config.safe_summary()["api_key"])

    def test_example_env_does_not_contain_real_secrets(self):
        example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

        self.assertIn("JOBPILOT_LLM_API_KEY=CHANGE_ME", example)
        self.assertIn("JOBPILOT_DATABASE_URL=mysql+pymysql://root:CHANGE_ME@", example)
        self.assertNotIn("123456", example)
        self.assertNotRegex(example, r"ghp_[A-Za-z0-9_]+")


if __name__ == "__main__":
    unittest.main()
