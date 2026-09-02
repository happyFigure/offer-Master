from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import (
    AppSettings,
    AuthSettings,
    ClaudeSettings,
    FeatureSettings,
    McpSettings,
    ProviderSettings,
    ServerSettings,
    SessionSettings,
    SkillSettings,
    WorkflowSettings,
)
from src.runtime_files import ensure_audit_links


class RuntimeFilesTests(unittest.TestCase):
    def test_ensure_audit_links_links_existing_claude_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            (config_dir / "sessions").mkdir(parents=True)
            (config_dir / "telemetry").mkdir()
            settings = _settings(root, config_dir)

            links = ensure_audit_links(settings)

            by_name = {item.name: item for item in links}
            self.assertEqual(by_name["claude-config"].status, "linked")
            self.assertEqual(by_name["claude-sessions"].status, "linked")
            self.assertEqual(by_name["claude-telemetry"].status, "linked")
            self.assertEqual(by_name["claude-logs"].status, "missing")
            self.assertTrue((root / "data" / "audit" / "claude-config").is_symlink())
            self.assertTrue((root / "data" / "audit" / "claude-sessions").is_symlink())
            self.assertTrue((root / "data" / "audit" / "audit-links.json").exists())

    def test_ensure_audit_links_removes_stale_missing_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / ".claude"
            config_dir.mkdir(parents=True)
            audit_dir = root / "data" / "audit"
            audit_dir.mkdir(parents=True)
            stale_target = root / "missing-sessions"
            stale_link = audit_dir / "claude-sessions"
            stale_link.symlink_to(stale_target, target_is_directory=True)
            settings = _settings(root, config_dir)

            links = ensure_audit_links(settings)

            by_name = {item.name: item for item in links}
            self.assertEqual(by_name["claude-sessions"].status, "missing")
            self.assertFalse(stale_link.exists())
            self.assertFalse(stale_link.is_symlink())


def _settings(root: Path, config_dir: Path) -> AppSettings:
    return AppSettings(
        root=root,
        server=ServerSettings(host="127.0.0.1", port=18008),
        claude=ClaudeSettings(
            workdir=root,
            config_dir=config_dir,
            default_model="MiniMax-M2.7",
            permission_mode="bypassPermissions",
            cli_path=None,
            setting_sources=["project", "local"],
            skills_filter="all",
            system_prompt_preset="claude_code",
            system_prompt_append="",
            system_prompt_file=None,
            include_hook_events=False,
            enable_file_checkpointing=False,
            attachment_text_char_limit=256000,
        ),
        provider=ProviderSettings(
            base_url="http://127.0.0.1:9999",
            anthropic_version="2023-06-01",
            api_key="",
            request_timeout_sec=60,
        ),
        auth=AuthSettings(
            enabled=False,
            uac_auth_url="",
            allow_users_path=root / "data" / "runtime" / "allow_users.json",
            shared_tdl_api_key_path=root / "allow_users.json",
        ),
        sessions=SessionSettings(
            mapping_path=root / "data" / "sessions" / "session-map.json",
            checkpoints_path=root / "data" / "sessions" / "checkpoints.json",
        ),
        skills=SkillSettings(source_dirs=[], mount_dir=root / "data" / "skill-mount"),
        workflows=WorkflowSettings(source_dirs=[], target_dir=config_dir / "workflows"),
        mcp=McpSettings(config_dir=root / "mcps", auto_load=False),
        features=FeatureSettings(
            auto_interrupt_on_disconnect=False,
            approval_frontend_enabled=False,
            question_frontend_enabled=False,
            hook_frontend_enabled=False,
            checkpoint_rewind_frontend_enabled=False,
            task_panel_frontend_enabled=False,
            subagent_events_frontend_enabled=False,
        ),
    )
