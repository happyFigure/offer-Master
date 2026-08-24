import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class RuntimeConfigurationTest(unittest.TestCase):
    def test_local_runtime_paths_stay_inside_project_root(self):
        from app.core.config import Settings

        settings = Settings(_env_file=None)

        self.assertEqual(PROJECT_ROOT / "logs", settings.log_dir)
        self.assertEqual(PROJECT_ROOT / "data" / "uploads", settings.uploads_path)
        self.assertEqual(PROJECT_ROOT / "data" / "imports", settings.imports_path)
        self.assertEqual(PROJECT_ROOT / "data" / "exports", settings.exports_path)
        self.assertNotEqual("c:", settings.log_dir.drive.lower())

    def test_feature_defaults_match_current_mvp_boundaries(self):
        from app.core.config import Settings

        settings = Settings(_env_file=None)

        self.assertEqual(["mock", "import_file"], settings.enabled_job_providers)
        self.assertFalse(settings.mcp_enabled)
        self.assertEqual("web_speech", settings.speech_provider)
        self.assertEqual(30, settings.worker_poll_interval_seconds)
        self.assertEqual(3, settings.worker_max_retries)
        self.assertFalse(settings.external_agent_auto_dispatch)
        self.assertFalse(settings.execution_planner_enabled)
        self.assertIsNone(settings.claude_sdk_agent_base_url)

    def test_claude_sdk_agent_executor_settings_can_be_configured(self):
        from app.core.config import Settings
        from app.agent_runtime.external_tasks.configured import _build_claude_sdk_http_executor_config

        with patch.dict(
            "os.environ",
            {
                "JOBPILOT_EXTERNAL_AGENT_AUTO_DISPATCH": "true",
                "JOBPILOT_CLAUDE_SDK_AGENT_BASE_URL": "http://127.0.0.1:18008/",
                "JOBPILOT_CLAUDE_SDK_AGENT_API_KEY": "tdl-key",
                "JOBPILOT_CLAUDE_SDK_AGENT_MODEL": "MiniMax-M2.7",
                "JOBPILOT_CLAUDE_SDK_AGENT_TIMEOUT_SECONDS": "9.5",
                "JOBPILOT_LLM_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "JOBPILOT_LLM_API_KEY": "sk-bailian",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        self.assertTrue(settings.external_agent_auto_dispatch)
        self.assertEqual("http://127.0.0.1:18008", settings.claude_sdk_agent_base_url)
        self.assertEqual("tdl-key", settings.claude_sdk_agent_api_key.get_secret_value())
        self.assertEqual("MiniMax-M2.7", settings.claude_sdk_agent_model)
        self.assertEqual(9.5, settings.claude_sdk_agent_timeout_seconds)
        executor_config = _build_claude_sdk_http_executor_config(settings, base_url=settings.claude_sdk_agent_base_url)
        self.assertEqual("https://dashscope.aliyuncs.com/apps/anthropic", executor_config.provider_base_url)
        self.assertEqual("sk-bailian", executor_config.provider_api_key)

    def test_intent_llm_settings_default_to_main_llm_and_can_be_overridden(self):
        from app.core.config import Settings
        from app.infrastructure.llm.client import build_intent_llm_runtime_config

        with patch.dict(
            "os.environ",
            {
                "JOBPILOT_LLM_PROVIDER": "bailian",
                "JOBPILOT_LLM_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "JOBPILOT_LLM_API_KEY": "sk-main",
                "JOBPILOT_LLM_MODEL": "qwen-plus",
                "JOBPILOT_INTENT_LLM_MODEL": "qwen-turbo",
                "JOBPILOT_INTENT_LLM_TIMEOUT_SECONDS": "12.5",
                "JOBPILOT_INTENT_LLM_MAX_RETRIES": "1",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        intent_config = build_intent_llm_runtime_config(settings)

        self.assertEqual("bailian", intent_config.provider)
        self.assertEqual("https://dashscope.aliyuncs.com/compatible-mode/v1", intent_config.base_url)
        self.assertEqual("sk-main", intent_config.api_key)
        self.assertEqual("qwen-turbo", intent_config.model)
        self.assertEqual(12.5, intent_config.timeout_seconds)
        self.assertEqual(1, intent_config.max_retries)

    def test_blank_intent_llm_api_key_reuses_main_llm_key(self):
        from app.core.config import Settings
        from app.infrastructure.llm.client import build_intent_llm_runtime_config

        with patch.dict(
            "os.environ",
            {
                "JOBPILOT_LLM_API_KEY": "sk-main",
                "JOBPILOT_INTENT_LLM_API_KEY": "",
                "JOBPILOT_INTENT_LLM_MODEL": "qwen-turbo",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        intent_config = build_intent_llm_runtime_config(settings)

        self.assertEqual("sk-main", intent_config.api_key)
        self.assertEqual("qwen-turbo", intent_config.model)

    def test_execution_planner_can_be_enabled_explicitly(self):
        from app.core.config import Settings

        with patch.dict("os.environ", {"JOBPILOT_EXECUTION_PLANNER_ENABLED": "true"}, clear=False):
            settings = Settings(_env_file=None)

        self.assertTrue(settings.execution_planner_enabled)

    def test_external_web_search_callback_prefers_claude_sdk_agent_when_configured(self):
        from app.core.config import Settings
        from app.agent_runtime.external_tasks.configured import build_external_web_search_callback

        with patch.dict(
            "os.environ",
            {
                "JOBPILOT_EXTERNAL_AGENT_AUTO_DISPATCH": "true",
                "JOBPILOT_CLAUDE_SDK_AGENT_BASE_URL": "http://127.0.0.1:18008/",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        class FakeClaudeSdkAdapter:
            def __init__(self, *, config):
                self.config = config

            def execute_web_search(self, query: str, *, max_results: int = 5):
                return {"executor_name": "claude-sdk-agent", "query": query, "answer": "ok", "sources": [], "max_results": max_results}

        with patch("app.agent_runtime.external_tasks.configured.ClaudeSdkHttpExecutorAdapter", FakeClaudeSdkAdapter), patch(
            "app.agent_runtime.external_tasks.configured._run_http_web_search"
        ) as fallback:
            callback = build_external_web_search_callback(settings)

            self.assertIsNotNone(callback)
            result = callback("梅西", 2)

        fallback.assert_not_called()
        self.assertEqual("claude-sdk-agent", result["executor_name"])
        self.assertEqual("ok", result["answer"])

    def test_external_web_search_callback_uses_http_fallback_without_claude_sdk_agent(self):
        from app.core.config import Settings
        from app.agent_runtime.external_tasks.configured import build_external_web_search_callback

        with patch.dict(
            "os.environ",
            {
                "JOBPILOT_EXTERNAL_AGENT_AUTO_DISPATCH": "true",
                "JOBPILOT_CLAUDE_SDK_AGENT_BASE_URL": "",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        with patch("app.agent_runtime.external_tasks.configured._run_http_web_search") as fallback:
            fallback.return_value = {"executor_name": "http-web-search-fallback", "query": "梅西", "answer": "fallback ok", "sources": []}
            callback = build_external_web_search_callback(settings)

            self.assertIsNotNone(callback)
            result = callback("梅西", 2)

        fallback.assert_called_once_with("梅西", max_results=2)
        self.assertEqual("http-web-search-fallback", result["executor_name"])
        self.assertEqual("fallback ok", result["answer"])

    def test_claude_sdk_agent_runtime_registration_uses_direct_executor_for_web_search(self):
        from app.agent_runtime.agent_as_tool import CLAUDE_SDK_AGENT_EXECUTOR_ID
        from app.agent_runtime.external_tasks.configured import build_agent_runtime_executor_bundle
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL
        from app.core.config import Settings

        with patch.dict(
            "os.environ",
            {
                "JOBPILOT_EXTERNAL_AGENT_AUTO_DISPATCH": "true",
                "JOBPILOT_CLAUDE_SDK_AGENT_BASE_URL": "http://127.0.0.1:18008/",
                "JOBPILOT_CLAUDE_SDK_AGENT_API_KEY": "tdl-key",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        executors, capability_executor_ids = build_agent_runtime_executor_bundle(settings)

        self.assertIn(CLAUDE_SDK_AGENT_EXECUTOR_ID, executors)
        self.assertEqual(CLAUDE_SDK_AGENT_EXECUTOR_ID, capability_executor_ids[EXTERNAL_WEB_SEARCH_TOOL])

    def test_xiaohongshu_rest_service_can_be_configured_without_c_drive_runtime_state(self):
        from app.core.config import Settings

        with patch.dict(
            "os.environ",
            {
                "JOBPILOT_XIAOHONGSHU_MCP_BASE_URL": "http://127.0.0.1:18060/",
                "JOBPILOT_XIAOHONGSHU_MCP_AUTH_TOKEN": "local-test-token",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        self.assertEqual("http://127.0.0.1:18060", settings.xiaohongshu_mcp_base_url)
        self.assertEqual("local-test-token", settings.xiaohongshu_mcp_auth_token.get_secret_value())
        self.assertNotEqual("c:", settings.imports_path.drive.lower())


if __name__ == "__main__":
    unittest.main()
