from __future__ import annotations

import unittest
from pathlib import Path

from src.config import ClaudeSettings
from src.permission_profile import (
    permission_options_from_runtime_config,
    permission_snapshot,
)


class PermissionProfileTests(unittest.TestCase):
    def test_request_config_resolves_agent_profile_without_writing_state(self) -> None:
        options = permission_options_from_runtime_config(
            _settings(),
            {
                "profile": "edit",
                "runtime_key": "agent-a",
                "revision": 4,
                "updated_at": 20,
                "updated_by": "u1",
            },
        )

        self.assertEqual(options.profile, "edit")
        self.assertEqual(options.permission_mode, "acceptEdits")
        self.assertEqual(options.runtime_key, "agent-a")
        self.assertEqual(options.revision, 4)
        self.assertEqual(options.updated_at, 20)
        self.assertEqual(options.updated_by, "u1")

    def test_native_bypass_keeps_agent_rules_and_full_bypass_clears_them(self) -> None:
        native = permission_options_from_runtime_config(
            _settings(),
            {"profile": "bypass", "runtimeKey": "agent-a"},
        )
        full = permission_options_from_runtime_config(
            _settings(),
            {"profile": "fullBypass", "runtimeKey": "agent-b"},
        )

        self.assertEqual(native.permission_mode, "bypassPermissions")
        self.assertEqual(native.disallowed_tools, ["Bash(rm -rf *)"])
        self.assertFalse(native.full_bypass)
        self.assertEqual(full.permission_mode, "bypassPermissions")
        self.assertIsNone(full.allowed_tools)
        self.assertIsNone(full.disallowed_tools)
        self.assertTrue(full.full_bypass)

    def test_invalid_or_missing_request_profile_falls_back_to_runtime_default(self) -> None:
        self.assertEqual(
            permission_options_from_runtime_config(_settings(), {}).profile,
            "safe",
        )
        self.assertEqual(
            permission_options_from_runtime_config(
                _settings(),
                {"profile": "not-supported"},
            ).profile,
            "safe",
        )

    def test_snapshot_has_no_sidecar_paths(self) -> None:
        snapshot = permission_snapshot(
            _settings(),
            {"profile": "fullBypass", "runtime_key": "agent-a", "revision": 2},
        )

        self.assertEqual(snapshot["scope"], "request")
        self.assertEqual(snapshot["current"]["profile"], "fullBypass")
        self.assertNotIn("storePath", snapshot)
        self.assertNotIn("legacyStorePath", snapshot)

    def test_readonly_compatibility_profile_exposes_native_plan_semantics(self) -> None:
        snapshot = permission_snapshot(_settings(), {"profile": "readonly"})
        plan_profile = next(item for item in snapshot["profiles"] if item["id"] == "readonly")

        self.assertEqual(snapshot["current"]["permissionMode"], "plan")
        self.assertEqual(plan_profile["title"], "规划模式")
        self.assertIn("ExitPlanMode", plan_profile["description"])


def _settings() -> ClaudeSettings:
    return ClaudeSettings(
        workdir=Path("/workspace"),
        config_dir=Path("/config"),
        default_model="test-model",
        permission_mode="default",
        cli_path=None,
        setting_sources=["user", "project", "local"],
        skills_filter="all",
        system_prompt_preset="claude_code",
        system_prompt_append="",
        system_prompt_file=None,
        include_hook_events=False,
        enable_file_checkpointing=False,
        attachment_text_char_limit=256000,
        allowed_tools=["Read"],
        disallowed_tools=["Bash(rm -rf *)"],
    )


if __name__ == "__main__":
    unittest.main()
