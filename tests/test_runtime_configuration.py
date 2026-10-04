import os
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

    def test_agent_context_window_defaults_to_current_qwen_plus_model_limit(self):
        from app.core.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        self.assertEqual("qwen-plus", settings.llm_model)
        self.assertEqual(1_000_000, settings.resolved_agent_context_window)

    def test_agent_context_window_can_be_overridden_explicitly(self):
        from app.core.config import Settings

        with patch.dict(os.environ, {"JOBPILOT_AGENT_CONTEXT_WINDOW": "128000"}, clear=True):
            settings = Settings(_env_file=None)

        self.assertEqual(128_000, settings.resolved_agent_context_window)

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

    def test_external_web_search_callback_prefers_bailian_search_when_enabled(self):
        from app.core.config import Settings
        from app.agent_runtime.external_tasks.configured import build_external_web_search_callback

        with patch.dict(
            "os.environ",
            {
                "JOBPILOT_EXTERNAL_AGENT_AUTO_DISPATCH": "true",
                "JOBPILOT_EXTERNAL_WEB_SEARCH_PROVIDER": "bailian",
                "JOBPILOT_CLAUDE_SDK_AGENT_BASE_URL": "http://127.0.0.1:18008/",
                "JOBPILOT_LLM_API_KEY": "sk-bailian",
                "JOBPILOT_LLM_MODEL": "qwen-plus",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        class FakeBailianSearchExecutor:
            def __init__(self, *, config):
                self.config = config

            def execute_web_search(self, query: str, *, max_results: int = 5):
                return {
                    "executor_name": "bailian-enable-search",
                    "query": query,
                    "answer": "百炼联网搜索结果",
                    "sources": ["https://example.com/search"],
                    "max_results": max_results,
                }

        with patch(
            "app.agent_runtime.external_tasks.configured.BailianWebSearchExecutor",
            FakeBailianSearchExecutor,
        ), patch("app.agent_runtime.external_tasks.configured.ClaudeSdkHttpExecutorAdapter") as claude_adapter:
            callback = build_external_web_search_callback(settings)

            self.assertIsNotNone(callback)
            result = callback("梅西最近比赛", 3)

        claude_adapter.assert_not_called()
        self.assertEqual("bailian-enable-search", result["executor_name"])
        self.assertEqual("百炼联网搜索结果", result["answer"])

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

    def test_bailian_web_search_provider_keeps_web_search_on_tool_registry(self):
        from app.agent_runtime.agent_as_tool import CLAUDE_SDK_AGENT_EXECUTOR_ID
        from app.agent_runtime.external_tasks.configured import build_agent_runtime_executor_bundle
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL
        from app.core.config import Settings

        with patch.dict(
            "os.environ",
            {
                "JOBPILOT_EXTERNAL_AGENT_AUTO_DISPATCH": "true",
                "JOBPILOT_EXTERNAL_WEB_SEARCH_PROVIDER": "bailian",
                "JOBPILOT_CLAUDE_SDK_AGENT_BASE_URL": "http://127.0.0.1:18008/",
                "JOBPILOT_CLAUDE_SDK_AGENT_API_KEY": "tdl-key",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        executors, capability_executor_ids = build_agent_runtime_executor_bundle(settings)

        self.assertNotIn(EXTERNAL_WEB_SEARCH_TOOL, capability_executor_ids)
        self.assertNotIn(CLAUDE_SDK_AGENT_EXECUTOR_ID, executors)

    def test_openai_sdk_agent_settings_can_be_configured(self):
        from app.core.config import Settings

        with patch.dict(
            "os.environ",
            {
                "JOBPILOT_EXTERNAL_AGENT_AUTO_DISPATCH": "true",
                "JOBPILOT_OPENAI_SDK_AGENT_ENABLED": "true",
                "JOBPILOT_OPENAI_SDK_AGENT_BASE_URL": "https://api.openai.example/v1/",
                "JOBPILOT_OPENAI_SDK_AGENT_API_KEY": "sk-openai-test",
                "JOBPILOT_OPENAI_SDK_AGENT_MODEL": "gpt-test",
                "JOBPILOT_OPENAI_SDK_AGENT_TIMEOUT_SECONDS": "33.5",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        self.assertTrue(settings.openai_sdk_agent_enabled)
        self.assertEqual("https://api.openai.example/v1", settings.openai_sdk_agent_base_url)
        self.assertEqual("sk-openai-test", settings.openai_sdk_agent_api_key.get_secret_value())
        self.assertEqual("gpt-test", settings.openai_sdk_agent_model)
        self.assertEqual(33.5, settings.openai_sdk_agent_timeout_seconds)

    def test_openai_sdk_agent_mode_defaults_to_legacy_chat_completions(self):
        from app.core.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        self.assertEqual("chat_completions", settings.openai_sdk_agent_mode)
        self.assertEqual(8, settings.sdk_agent_max_turns)
        self.assertFalse(settings.sdk_agent_enable_web_research)
        self.assertFalse(settings.sdk_agent_enable_file_analysis)
        self.assertFalse(settings.sdk_agent_enable_mutation_tools)
        self.assertTrue(settings.sdk_agent_require_approval_for_mutation)
        self.assertEqual("none", settings.sdk_agent_sandbox_mode)

    def test_openai_sdk_agent_runner_settings_can_be_configured(self):
        from app.core.config import Settings

        with patch.dict(
            os.environ,
            {
                "JOBPILOT_OPENAI_SDK_AGENT_MODE": "agents_sdk",
                "JOBPILOT_SDK_AGENT_MAX_TURNS": "12",
                "JOBPILOT_SDK_AGENT_ENABLE_WEB_RESEARCH": "true",
                "JOBPILOT_SDK_AGENT_ENABLE_FILE_ANALYSIS": "true",
                "JOBPILOT_SDK_AGENT_ENABLE_MUTATION_TOOLS": "true",
                "JOBPILOT_SDK_AGENT_REQUIRE_APPROVAL_FOR_MUTATION": "false",
                "JOBPILOT_SDK_AGENT_SANDBOX_MODE": "worktree",
            },
            clear=True,
        ):
            settings = Settings(_env_file=None)

        self.assertEqual("agents_sdk", settings.openai_sdk_agent_mode)
        self.assertEqual(12, settings.sdk_agent_max_turns)
        self.assertTrue(settings.sdk_agent_enable_web_research)
        self.assertTrue(settings.sdk_agent_enable_file_analysis)
        self.assertTrue(settings.sdk_agent_enable_mutation_tools)
        self.assertFalse(settings.sdk_agent_require_approval_for_mutation)
        self.assertEqual("worktree", settings.sdk_agent_sandbox_mode)

    def test_openai_sdk_agent_config_falls_back_to_main_llm_settings(self):
        from app.agent_runtime.external_tasks.configured import _build_openai_sdk_agent_config
        from app.core.config import Settings

        with patch.dict(
            "os.environ",
            {
                "JOBPILOT_OPENAI_SDK_AGENT_ENABLED": "true",
                "JOBPILOT_OPENAI_SDK_AGENT_BASE_URL": "",
                "JOBPILOT_OPENAI_SDK_AGENT_API_KEY": "",
                "JOBPILOT_OPENAI_SDK_AGENT_MODEL": "",
                "JOBPILOT_LLM_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "JOBPILOT_LLM_API_KEY": "sk-main-model",
                "JOBPILOT_LLM_MODEL": "qwen-plus",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        config = _build_openai_sdk_agent_config(settings)

        self.assertEqual("https://dashscope.aliyuncs.com/compatible-mode/v1", config.base_url)
        self.assertEqual("sk-main-model", config.api_key)
        self.assertEqual("qwen-plus", config.model)

    def test_openai_sdk_agent_runtime_registration_adds_resume_tailoring_capability(self):
        from app.agent_runtime.agent_as_tool import CLAUDE_SDK_AGENT_EXECUTOR_ID, OPENAI_SDK_AGENT_EXECUTOR_ID
        from app.agent_runtime.external_tasks.configured import build_agent_runtime_executor_bundle
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL
        from app.core.config import Settings

        with patch.dict(
            "os.environ",
            {
                "JOBPILOT_EXTERNAL_AGENT_AUTO_DISPATCH": "true",
                "JOBPILOT_EXTERNAL_WEB_SEARCH_PROVIDER": "bailian",
                "JOBPILOT_CLAUDE_SDK_AGENT_BASE_URL": "http://127.0.0.1:18008/",
                "JOBPILOT_OPENAI_SDK_AGENT_ENABLED": "true",
                "JOBPILOT_OPENAI_SDK_AGENT_API_KEY": "sk-openai-test",
                "JOBPILOT_OPENAI_SDK_AGENT_MODEL": "gpt-test",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        executors, capability_executor_ids = build_agent_runtime_executor_bundle(settings)

        self.assertIn(OPENAI_SDK_AGENT_EXECUTOR_ID, executors)
        self.assertEqual(OPENAI_SDK_AGENT_EXECUTOR_ID, capability_executor_ids["resume.tailor"])
        self.assertNotIn(EXTERNAL_WEB_SEARCH_TOOL, capability_executor_ids)
        self.assertNotIn(CLAUDE_SDK_AGENT_EXECUTOR_ID, executors)

    def test_openai_sdk_agent_runtime_registration_uses_main_llm_key_when_agent_key_is_blank(self):
        from app.agent_runtime.agent_as_tool import OPENAI_SDK_AGENT_EXECUTOR_ID
        from app.agent_runtime.external_tasks.configured import build_agent_runtime_executor_bundle
        from app.core.config import Settings

        with patch.dict(
            "os.environ",
            {
                "JOBPILOT_EXTERNAL_AGENT_AUTO_DISPATCH": "true",
                "JOBPILOT_EXTERNAL_WEB_SEARCH_PROVIDER": "bailian",
                "JOBPILOT_OPENAI_SDK_AGENT_ENABLED": "true",
                "JOBPILOT_OPENAI_SDK_AGENT_API_KEY": "",
                "JOBPILOT_OPENAI_SDK_AGENT_MODEL": "",
                "JOBPILOT_OPENAI_SDK_AGENT_BASE_URL": "",
                "JOBPILOT_LLM_API_KEY": "sk-main-model",
                "JOBPILOT_LLM_MODEL": "qwen-plus",
                "JOBPILOT_LLM_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        executors, capability_executor_ids = build_agent_runtime_executor_bundle(settings)

        self.assertIn(OPENAI_SDK_AGENT_EXECUTOR_ID, executors)
        self.assertEqual(OPENAI_SDK_AGENT_EXECUTOR_ID, capability_executor_ids["resume.tailor"])

    def test_agents_sdk_mode_registers_openai_runner_executor(self):
        from app.agent_runtime.agent_as_tool import OPENAI_SDK_AGENT_EXECUTOR_ID
        from app.agent_runtime.external_tasks.configured import build_agent_runtime_executor_bundle
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerExecutor
        from app.core.config import Settings

        with patch.dict(
            os.environ,
            {
                "JOBPILOT_EXTERNAL_AGENT_AUTO_DISPATCH": "true",
                "JOBPILOT_EXTERNAL_WEB_SEARCH_PROVIDER": "bailian",
                "JOBPILOT_OPENAI_SDK_AGENT_ENABLED": "true",
                "JOBPILOT_OPENAI_SDK_AGENT_MODE": "agents_sdk",
                "JOBPILOT_OPENAI_SDK_AGENT_API_KEY": "sk-openai-test",
                "JOBPILOT_OPENAI_SDK_AGENT_MODEL": "gpt-test",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        executors, capability_executor_ids = build_agent_runtime_executor_bundle(settings)

        self.assertIn(OPENAI_SDK_AGENT_EXECUTOR_ID, executors)
        self.assertIsInstance(executors[OPENAI_SDK_AGENT_EXECUTOR_ID], OpenAISdkAgentRunnerExecutor)
        self.assertEqual(OPENAI_SDK_AGENT_EXECUTOR_ID, capability_executor_ids["resume.tailor"])

    def test_agents_sdk_mode_can_downshift_web_search_to_openai_runner_executor(self):
        from app.agent_runtime.agent_as_tool import OPENAI_SDK_AGENT_EXECUTOR_ID
        from app.agent_runtime.external_tasks.configured import build_agent_runtime_executor_bundle
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerExecutor
        from app.agent_runtime.tool_registry import EXTERNAL_WEB_SEARCH_TOOL, AgentToolDefinition, AgentToolRegistry
        from app.core.config import Settings

        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name=EXTERNAL_WEB_SEARCH_TOOL,
                    description="Search the public web.",
                    input_schema={"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"tool_name": EXTERNAL_WEB_SEARCH_TOOL, "ok": True, "result": {}},
                )
            ]
        )

        with patch.dict(
            os.environ,
            {
                "JOBPILOT_EXTERNAL_AGENT_AUTO_DISPATCH": "true",
                "JOBPILOT_EXTERNAL_WEB_SEARCH_PROVIDER": "bailian",
                "JOBPILOT_OPENAI_SDK_AGENT_ENABLED": "true",
                "JOBPILOT_OPENAI_SDK_AGENT_MODE": "agents_sdk",
                "JOBPILOT_OPENAI_SDK_AGENT_API_KEY": "sk-openai-test",
                "JOBPILOT_OPENAI_SDK_AGENT_MODEL": "gpt-test",
                "JOBPILOT_SDK_AGENT_ENABLE_WEB_RESEARCH": "true",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        executors, capability_executor_ids = build_agent_runtime_executor_bundle(
            settings,
            tool_registry=registry,
            session_provider=lambda _context: "db-session",
        )

        self.assertIn(OPENAI_SDK_AGENT_EXECUTOR_ID, executors)
        self.assertIsInstance(executors[OPENAI_SDK_AGENT_EXECUTOR_ID], OpenAISdkAgentRunnerExecutor)
        self.assertEqual(OPENAI_SDK_AGENT_EXECUTOR_ID, capability_executor_ids["resume.tailor"])
        self.assertEqual(OPENAI_SDK_AGENT_EXECUTOR_ID, capability_executor_ids[EXTERNAL_WEB_SEARCH_TOOL])

    def test_agents_sdk_mode_allows_mutation_tools_only_with_temp_copy_sandbox(self):
        from app.agent_runtime.agent_as_tool import FILESYSTEM_SKILL_CAPABILITY, OPENAI_SDK_AGENT_EXECUTOR_ID
        from app.agent_runtime.external_tasks.configured import build_agent_runtime_executor_bundle
        from app.agent_runtime.sdk_agents.runner_adapter import OpenAISdkAgentRunnerExecutor
        from app.agent_runtime.tool_registry import (
            FILESYSTEM_READ_FILE_TOOL,
            FILESYSTEM_WRITE_TEXT_TOOL,
            SKILL_LIST_ACTIONS_TOOL,
            SKILL_LIST_TOOL,
            SKILL_READ_TOOL,
            AgentToolDefinition,
            AgentToolRegistry,
            AgentToolRiskLevel,
            create_skill_lazy_agent_tool_definitions,
        )
        from app.core.config import Settings

        registry = AgentToolRegistry(
            create_skill_lazy_agent_tool_definitions()
            + [
                AgentToolDefinition(
                    name=FILESYSTEM_READ_FILE_TOOL,
                    description="Read sandbox input.",
                    input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"tool_name": FILESYSTEM_READ_FILE_TOOL, "ok": True},
                    risk_level=AgentToolRiskLevel.LOW,
                    requires_confirmation=False,
                ),
                AgentToolDefinition(
                    name=FILESYSTEM_WRITE_TEXT_TOOL,
                    description="Write sandbox output.",
                    input_schema={"type": "object", "required": ["path", "text"], "properties": {"path": {"type": "string"}}},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"tool_name": FILESYSTEM_WRITE_TEXT_TOOL, "ok": True},
                    risk_level=AgentToolRiskLevel.HIGH,
                    requires_confirmation=True,
                ),
            ]
        )

        with patch.dict(
            os.environ,
            {
                "JOBPILOT_EXTERNAL_AGENT_AUTO_DISPATCH": "true",
                "JOBPILOT_EXTERNAL_WEB_SEARCH_PROVIDER": "bailian",
                "JOBPILOT_OPENAI_SDK_AGENT_ENABLED": "true",
                "JOBPILOT_OPENAI_SDK_AGENT_MODE": "agents_sdk",
                "JOBPILOT_OPENAI_SDK_AGENT_API_KEY": "sk-openai-test",
                "JOBPILOT_OPENAI_SDK_AGENT_MODEL": "gpt-test",
                "JOBPILOT_SDK_AGENT_ENABLE_FILE_ANALYSIS": "true",
                "JOBPILOT_SDK_AGENT_ENABLE_MUTATION_TOOLS": "true",
                "JOBPILOT_SDK_AGENT_SANDBOX_MODE": "none",
            },
            clear=False,
        ):
            settings_without_sandbox = Settings(_env_file=None)

        without_sandbox, _ = build_agent_runtime_executor_bundle(settings_without_sandbox, tool_registry=registry)
        adapter_without_sandbox = without_sandbox[OPENAI_SDK_AGENT_EXECUTOR_ID]._adapter
        self.assertIn(FILESYSTEM_READ_FILE_TOOL, adapter_without_sandbox.allowed_tools)
        self.assertIn(SKILL_LIST_TOOL, adapter_without_sandbox.allowed_tools)
        self.assertIn(SKILL_LIST_ACTIONS_TOOL, adapter_without_sandbox.allowed_tools)
        self.assertIn(SKILL_READ_TOOL, adapter_without_sandbox.allowed_tools)
        self.assertNotIn(FILESYSTEM_WRITE_TEXT_TOOL, adapter_without_sandbox.allowed_tools)

        with patch.dict(
            os.environ,
            {
                "JOBPILOT_EXTERNAL_AGENT_AUTO_DISPATCH": "true",
                "JOBPILOT_EXTERNAL_WEB_SEARCH_PROVIDER": "bailian",
                "JOBPILOT_OPENAI_SDK_AGENT_ENABLED": "true",
                "JOBPILOT_OPENAI_SDK_AGENT_MODE": "agents_sdk",
                "JOBPILOT_OPENAI_SDK_AGENT_API_KEY": "sk-openai-test",
                "JOBPILOT_OPENAI_SDK_AGENT_MODEL": "gpt-test",
                "JOBPILOT_SDK_AGENT_ENABLE_FILE_ANALYSIS": "true",
                "JOBPILOT_SDK_AGENT_ENABLE_MUTATION_TOOLS": "true",
                "JOBPILOT_SDK_AGENT_SANDBOX_MODE": "temp_copy",
            },
            clear=False,
        ):
            settings_with_sandbox = Settings(_env_file=None)

        with_sandbox, capability_executor_ids = build_agent_runtime_executor_bundle(settings_with_sandbox, tool_registry=registry)
        executor = with_sandbox[OPENAI_SDK_AGENT_EXECUTOR_ID]

        self.assertIsInstance(executor, OpenAISdkAgentRunnerExecutor)
        self.assertIn(FILESYSTEM_READ_FILE_TOOL, executor._adapter.allowed_tools)
        self.assertIn(FILESYSTEM_WRITE_TEXT_TOOL, executor._adapter.allowed_tools)
        self.assertEqual(OPENAI_SDK_AGENT_EXECUTOR_ID, capability_executor_ids[FILESYSTEM_SKILL_CAPABILITY])
        self.assertNotIn(FILESYSTEM_WRITE_TEXT_TOOL, capability_executor_ids)
        self.assertEqual(settings_with_sandbox.exports_path.parent / "sdk-agent-runs", Path(executor._adapter.sandbox_manager.base_dir))

    def test_agents_sdk_mode_isolates_chrome_and_dbx_tools_behind_child_agents(self):
        from app.agent_runtime.agent_as_tool import OPENAI_SDK_AGENT_EXECUTOR_ID
        from app.agent_runtime.external_tasks.configured import build_agent_runtime_executor_bundle
        from app.agent_runtime.tool_registry import AgentToolDefinition, AgentToolRegistry, AgentToolRiskLevel
        from app.core.config import Settings

        chrome_tool = "mcp.chrome.list_pages"
        dbx_tool = "mcp.dbx.dbx_list_connections"
        registry = AgentToolRegistry(
            [
                AgentToolDefinition(
                    name=chrome_tool,
                    description="List pages from Google Chrome MCP.",
                    input_schema={"type": "object", "additionalProperties": True},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"tool_name": chrome_tool, "ok": True},
                    risk_level=AgentToolRiskLevel.LOW,
                    requires_confirmation=False,
                ),
                AgentToolDefinition(
                    name=dbx_tool,
                    description="List DBX connections from dbx MCP.",
                    input_schema={"type": "object", "additionalProperties": True},
                    output_schema={"type": "object"},
                    handler=lambda _session, **_arguments: {"tool_name": dbx_tool, "ok": True},
                    risk_level=AgentToolRiskLevel.LOW,
                    requires_confirmation=False,
                ),
            ]
        )

        with patch.dict(
            os.environ,
            {
                "JOBPILOT_EXTERNAL_AGENT_AUTO_DISPATCH": "true",
                "JOBPILOT_EXTERNAL_WEB_SEARCH_PROVIDER": "bailian",
                "JOBPILOT_OPENAI_SDK_AGENT_ENABLED": "true",
                "JOBPILOT_OPENAI_SDK_AGENT_MODE": "agents_sdk",
                "JOBPILOT_OPENAI_SDK_AGENT_API_KEY": "sk-openai-test",
                "JOBPILOT_OPENAI_SDK_AGENT_MODEL": "gpt-test",
                "JOBPILOT_SDK_AGENT_ENABLE_CHROME_MCP": "true",
                "JOBPILOT_SDK_AGENT_ENABLE_DBX_MCP": "true",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        executors, capability_executor_ids = build_agent_runtime_executor_bundle(settings, tool_registry=registry)
        adapter = executors[OPENAI_SDK_AGENT_EXECUTOR_ID]._adapter

        self.assertIn(chrome_tool, adapter.allowed_tools)
        self.assertIn(dbx_tool, adapter.allowed_tools)
        self.assertNotIn(chrome_tool, adapter.risk_policy["delegation"]["exposed_capabilities"])
        self.assertNotIn(dbx_tool, adapter.risk_policy["delegation"]["exposed_capabilities"])
        self.assertEqual(
            [chrome_tool],
            adapter.risk_policy["delegation"]["tools_by_capability"]["agent.google_chrome"],
        )
        self.assertEqual(
            [dbx_tool],
            adapter.risk_policy["delegation"]["tools_by_capability"]["agent.dbx_readonly"],
        )
        self.assertNotIn(chrome_tool, capability_executor_ids)
        self.assertNotIn(dbx_tool, capability_executor_ids)
        self.assertEqual(OPENAI_SDK_AGENT_EXECUTOR_ID, capability_executor_ids["agent.google_chrome"])
        self.assertEqual(OPENAI_SDK_AGENT_EXECUTOR_ID, capability_executor_ids["agent.dbx_readonly"])

    def test_sdk_agent_mcp_flags_extend_gateway_allowlist_for_chrome_and_dbx(self):
        from app.core.config import Settings

        with patch.dict(
            os.environ,
            {
                "JOBPILOT_MCP_TOOL_ALLOWLIST": "open_page",
                "JOBPILOT_SDK_AGENT_ENABLE_CHROME_MCP": "true",
                "JOBPILOT_SDK_AGENT_ENABLE_DBX_MCP": "true",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        self.assertEqual(
            [
                "open_page",
                "chrome.list_pages",
                "dbx.dbx_list_connections",
                "dbx.dbx_list_databases",
                "dbx.dbx_list_tables",
                "dbx.dbx_get_schema_context",
                "dbx.dbx_describe_table",
                "dbx.dbx_execute_query",
            ],
            settings.allowed_mcp_tools,
        )

    def test_dbx_allowlist_drops_non_readonly_tools(self):
        from app.core.config import Settings

        with patch.dict(
            os.environ,
            {
                "JOBPILOT_MCP_TOOL_ALLOWLIST": "dbx.dbx_commit,dbx.dbx_execute_batch,dbx.dbx_execute_query",
                "JOBPILOT_SDK_AGENT_ENABLE_DBX_MCP": "true",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        self.assertIn("dbx.dbx_execute_query", settings.allowed_mcp_tools)
        self.assertNotIn("dbx.dbx_commit", settings.allowed_mcp_tools)
        self.assertNotIn("dbx.dbx_execute_batch", settings.allowed_mcp_tools)

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
