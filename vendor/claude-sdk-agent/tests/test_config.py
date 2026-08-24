from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import _claude_cli_version, _load_claude_settings_json, _resolve_cli_path, load_settings


class ConfigTests(unittest.TestCase):
    def test_load_settings_uses_large_default_attachment_text_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "service.json").write_text(
                json.dumps(
                    {
                        "server": {"host": "127.0.0.1", "port": 18008},
                        "claude": {"config_dir": ".", "workdir": "."},
                        "provider": {"base_url": "http://127.0.0.1:9999"},
                        "mcp": {"config_dir": "mcps", "extra_config_dirs": ["private-mcps"]},
                    }
                ),
                encoding="utf-8",
            )

            settings = load_settings(root)

            self.assertEqual(settings.claude.attachment_text_char_limit, 256000)

    def test_load_settings_reads_claude_sdk_passthrough_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "service.json").write_text(
                json.dumps(
                    {
                        "server": {"host": "127.0.0.1", "port": 18008},
                        "claude": {
                            "config_dir": ".",
                            "workdir": ".",
                            "tools": {"type": "preset", "preset": "claude_code"},
                            "allowed_tools": ["Read", "mcp__demo__*"],
                            "disallowed_tools": ["Bash"],
                            "strict_mcp_config": True,
                            "continue_conversation": True,
                            "max_turns": 7,
                            "max_budget_usd": 0.25,
                            "fallback_model": "MiniMax-Fallback",
                            "betas": ["context-1m-2025-08-07"],
                            "permission_prompt_tool_name": "mcp__approval__ask",
                            "settings": "config/claude-settings.json",
                            "add_dirs": ["shared"],
                            "env": {"CLAUDE_AGENT_SDK_CLIENT_APP": "test-app/1.0"},
                            "extra_args": {"verbose": None, "debug": "1"},
                            "max_buffer_size": 12345,
                            "user": "user-1",
                            "include_partial_messages": False,
                            "fork_session": True,
                            "agents": {"reviewer": {"description": "Review code", "prompt": "Be strict"}},
                            "sandbox": {"enabled": False},
                            "plugins": [{"type": "local", "path": "./plugins/demo"}],
                            "max_thinking_tokens": 4096,
                            "thinking": {"type": "adaptive"},
                            "effort": "high",
                            "output_format": {"type": "json_schema", "schema": {"type": "object"}},
                            "session_store_flush": "eager",
                            "load_timeout_ms": 1000,
                            "task_budget": {"total": 10000},
                        },
                        "provider": {"base_url": "http://127.0.0.1:9999"},
                        "skill_usage_audit": {
                            "enabled": True,
                            "base_url": "http://127.0.0.1:18000/",
                            "endpoint": "v1/skill-center/usage",
                            "timeout_sec": 2.5,
                        },
                        "workflows": {"source_dirs": ["workflows"], "target_dir": ".claude/workflows"},
                        "mcp": {"config_dir": "mcps", "extra_config_dirs": ["private-mcps"]},
                    }
                ),
                encoding="utf-8",
            )

            settings = load_settings(root)

            self.assertEqual(settings.claude.tools, {"type": "preset", "preset": "claude_code"})
            self.assertEqual(settings.claude.allowed_tools, ["Read", "mcp__demo__*"])
            self.assertEqual(settings.claude.disallowed_tools, ["Bash"])
            self.assertTrue(settings.claude.strict_mcp_config)
            self.assertTrue(settings.claude.continue_conversation)
            self.assertEqual(settings.claude.max_turns, 7)
            self.assertEqual(settings.claude.max_budget_usd, 0.25)
            self.assertEqual(settings.claude.fallback_model, "MiniMax-Fallback")
            self.assertEqual(settings.claude.betas, ["context-1m-2025-08-07"])
            self.assertEqual(settings.claude.permission_prompt_tool_name, "mcp__approval__ask")
            self.assertEqual(Path(settings.claude.settings).name, "claude-settings.json")
            self.assertEqual(settings.claude.add_dirs, [root / "shared"])
            self.assertEqual(settings.claude.env, {"CLAUDE_AGENT_SDK_CLIENT_APP": "test-app/1.0"})
            self.assertEqual(settings.claude.extra_args, {"verbose": None, "debug": "1"})
            self.assertEqual(settings.claude.max_buffer_size, 12345)
            self.assertEqual(settings.claude.user, "user-1")
            self.assertFalse(settings.claude.include_partial_messages)
            self.assertTrue(settings.claude.fork_session)
            self.assertEqual(settings.claude.agents["reviewer"]["prompt"], "Be strict")
            self.assertEqual(settings.claude.sandbox, {"enabled": False})
            self.assertEqual(settings.claude.plugins, [{"type": "local", "path": "./plugins/demo"}])
            self.assertEqual(settings.claude.max_thinking_tokens, 4096)
            self.assertEqual(settings.claude.thinking, {"type": "adaptive"})
            self.assertEqual(settings.claude.effort, "high")
            self.assertEqual(settings.claude.output_format, {"type": "json_schema", "schema": {"type": "object"}})
            self.assertEqual(settings.claude.session_store_flush, "eager")
            self.assertEqual(settings.claude.load_timeout_ms, 1000)
            self.assertEqual(settings.claude.task_budget, {"total": 10000})
            self.assertTrue(settings.skill_usage_audit.enabled)
            self.assertEqual(settings.skill_usage_audit.base_url, "http://127.0.0.1:18000")
            self.assertEqual(settings.skill_usage_audit.endpoint, "v1/skill-center/usage")
            self.assertEqual(settings.skill_usage_audit.timeout_sec, 2.5)
            self.assertEqual(settings.workflows.source_dirs, [root / "workflows"])
            self.assertEqual(settings.workflows.target_dir, root / ".claude" / "workflows")
            self.assertEqual(settings.mcp.config_dir, root / "mcps")
            self.assertEqual(settings.mcp.extra_config_dirs, [root / "private-mcps"])

    def test_load_claude_settings_json_bootstraps_settings_from_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / ".claude"
            config_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "model": "MiniMax-M2.7",
                "env": {
                    "ANTHROPIC_BASE_URL": "https://example.invalid",
                },
            }
            (config_dir / "settings.template.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            data = _load_claude_settings_json(config_dir)

            self.assertEqual(data, payload)
            self.assertTrue((config_dir / "settings.json").exists())
            self.assertEqual(
                json.loads((config_dir / "settings.json").read_text(encoding="utf-8")),
                payload,
            )

    def test_resolve_cli_path_uses_home_candidates_when_path_lookup_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            candidate = home / ".npm-global" / "bin" / "claude"
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text("#!/bin/sh\necho '2.1.198 (Claude Code)'\n", encoding="utf-8")
            candidate.chmod(0o755)
            with patch("src.config.shutil.which", return_value=None), patch("src.config.Path.home", return_value=home):
                self.assertEqual(_resolve_cli_path(""), str(candidate))

    def test_resolve_cli_path_ignores_existing_but_unrunnable_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            candidate = home / ".npm-global" / "bin" / "claude"
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text("#!/usr/bin/env missing-claude-runtime\n", encoding="utf-8")
            candidate.chmod(0o755)

            with patch("src.config.shutil.which", return_value=None), patch("src.config.Path.home", return_value=home):
                self.assertIsNone(_resolve_cli_path(""))

    def test_resolve_cli_path_selects_highest_version_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            old_cli = home / ".local" / "opt" / "node-v18.20.1-linux-x64" / "bin" / "claude"
            new_cli = home / ".local" / "opt" / "node-v22.23.1-linux-x64" / "bin" / "claude"
            old_cli.parent.mkdir(parents=True, exist_ok=True)
            new_cli.parent.mkdir(parents=True, exist_ok=True)
            old_cli.write_text("#!/bin/sh\necho '2.1.132 (Claude Code)'\n", encoding="utf-8")
            new_cli.write_text("#!/bin/sh\necho '2.1.198 (Claude Code)'\n", encoding="utf-8")
            old_cli.chmod(0o755)
            new_cli.chmod(0o755)

            with patch("src.config.shutil.which", return_value=str(old_cli)), patch("src.config.Path.home", return_value=home):
                self.assertEqual(_resolve_cli_path(""), str(new_cli))

    def test_claude_cli_version_parses_claude_code_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cli = Path(tmp) / "claude"
            cli.write_text("#!/bin/sh\necho '2.1.198 (Claude Code)'\n", encoding="utf-8")
            cli.chmod(0o755)

            self.assertEqual(_claude_cli_version(str(cli)), (2, 1, 198))

    def test_claude_cli_version_rejects_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cli = Path(tmp) / "claude"
            cli.write_text("#!/bin/sh\necho '2.1.198 unavailable'\nexit 127\n", encoding="utf-8")
            cli.chmod(0o755)

            self.assertEqual(_claude_cli_version(str(cli)), ())
