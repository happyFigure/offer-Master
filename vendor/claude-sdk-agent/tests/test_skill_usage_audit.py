from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.config import SkillUsageAuditSettings
from src.skill_usage_audit import SkillUsageAuditor


class _FakeAsyncClient:
    calls: list[dict[str, object]] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, object], headers: dict[str, str]):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return SimpleNamespace(status_code=200, text="{}", json=lambda: {"ok": True})


class SkillUsageAuditorTests(unittest.IsolatedAsyncioTestCase):
    async def test_records_successful_bash_script_under_mounted_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "source" / "demo-skill"
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("---\nname: demo-skill\n---\n", encoding="utf-8")
            script_path = scripts_dir / "run_demo.py"
            script_path.write_text("print('ok')\n", encoding="utf-8")
            mount_skill_dir = root / "mount" / ".claude" / "skills" / "demo-skill"
            mount_skill_dir.parent.mkdir(parents=True)
            mount_skill_dir.symlink_to(skill_dir, target_is_directory=True)

            _FakeAsyncClient.calls = []
            auditor = SkillUsageAuditor(
                settings=SkillUsageAuditSettings(
                    enabled=True,
                    base_url="http://127.0.0.1:18000",
                    endpoint="/v1/skill-center/usage",
                    timeout_sec=3.0,
                ),
                skill_mount_root=root / "mount",
                base_context={"operator_user_id": "10358560", "entry": "chat"},
                request_headers={"x-session-id": "sess-1"},
            )

            with patch("src.skill_usage_audit.httpx.AsyncClient", _FakeAsyncClient):
                auditor.observe_tool_start(
                    {
                        "toolCallId": "call-1",
                        "name": "Bash",
                        "arguments": {"command": f"python3.11 {mount_skill_dir / 'scripts' / 'run_demo.py'}"},
                    },
                    cwd=root,
                )
                auditor.observe_tool_result({"toolCallId": "call-1", "status": "completed"})
                await asyncio.gather(*list(auditor._tasks))

        self.assertEqual(len(_FakeAsyncClient.calls), 1)
        call = _FakeAsyncClient.calls[0]
        self.assertEqual(call["url"], "http://127.0.0.1:18000/v1/skill-center/usage")
        payload = call["json"]
        self.assertEqual(payload["skillDir"], str(skill_dir.resolve()))
        audit_context = payload["auditContext"]
        self.assertEqual(audit_context["operator_user_id"], "10358560")
        self.assertEqual(audit_context["skill_action"], "run_demo")
        self.assertEqual(audit_context["skill_invocation"], "claude_code_bash_attributed")
        self.assertEqual(audit_context["runtime"], "claude-code")
        self.assertEqual(call["headers"]["x-session-id"], "sess-1")

    async def test_records_direct_workflow_skill_command_without_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "source" / "workflow-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("---\nname: workflow-skill\n---\nRun workflow.", encoding="utf-8")
            mount_skill_dir = root / "mount" / ".claude" / "skills" / "workflow-skill"
            mount_skill_dir.parent.mkdir(parents=True)
            mount_skill_dir.symlink_to(skill_dir, target_is_directory=True)

            _FakeAsyncClient.calls = []
            auditor = SkillUsageAuditor(
                settings=SkillUsageAuditSettings(
                    enabled=True,
                    base_url="http://127.0.0.1:18000",
                    endpoint="/v1/skill-center/usage",
                    timeout_sec=3.0,
                ),
                skill_mount_root=root / "mount",
                base_context={"operator_user_id": "10358560"},
                request_headers={},
            )

            with patch("src.skill_usage_audit.httpx.AsyncClient", _FakeAsyncClient):
                auditor.record_workflow_skill_command("workflow-skill")
                await asyncio.gather(*list(auditor._tasks))

        self.assertEqual(len(_FakeAsyncClient.calls), 1)
        payload = _FakeAsyncClient.calls[0]["json"]
        self.assertEqual(payload["skillDir"], str(skill_dir.resolve()))
        audit_context = payload["auditContext"]
        self.assertEqual(audit_context["skill_action"], "workflow")
        self.assertEqual(audit_context["skill_invocation"], "claude_code_skill_command_workflow")

    async def test_direct_skill_command_does_not_record_script_skill_before_bash_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "source" / "script-skill"
            (skill_dir / "scripts").mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("---\nname: script-skill\n---\n", encoding="utf-8")
            mount_skill_dir = root / "mount" / ".claude" / "skills" / "script-skill"
            mount_skill_dir.parent.mkdir(parents=True)
            mount_skill_dir.symlink_to(skill_dir, target_is_directory=True)

            _FakeAsyncClient.calls = []
            auditor = SkillUsageAuditor(
                settings=SkillUsageAuditSettings(
                    enabled=True,
                    base_url="http://127.0.0.1:18000",
                    endpoint="/v1/skill-center/usage",
                    timeout_sec=3.0,
                ),
                skill_mount_root=root / "mount",
                base_context={},
                request_headers={},
            )

            with patch("src.skill_usage_audit.httpx.AsyncClient", _FakeAsyncClient):
                auditor.record_workflow_skill_command("script-skill")
                await asyncio.sleep(0)

        self.assertEqual(_FakeAsyncClient.calls, [])

    async def test_does_not_record_failed_bash_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "source" / "demo-skill"
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("---\nname: demo-skill\n---\n", encoding="utf-8")
            script_path = scripts_dir / "fail_demo.py"
            script_path.write_text("raise SystemExit(2)\n", encoding="utf-8")
            mount_skill_dir = root / "mount" / ".claude" / "skills" / "demo-skill"
            mount_skill_dir.parent.mkdir(parents=True)
            mount_skill_dir.symlink_to(skill_dir, target_is_directory=True)

            _FakeAsyncClient.calls = []
            auditor = SkillUsageAuditor(
                settings=SkillUsageAuditSettings(
                    enabled=True,
                    base_url="http://127.0.0.1:18000",
                    endpoint="/v1/skill-center/usage",
                    timeout_sec=3.0,
                ),
                skill_mount_root=root / "mount",
                base_context={},
                request_headers={},
            )

            with patch("src.skill_usage_audit.httpx.AsyncClient", _FakeAsyncClient):
                auditor.observe_tool_start(
                    {
                        "toolCallId": "call-1",
                        "name": "Bash",
                        "arguments": {"command": f"python {mount_skill_dir / 'scripts' / 'fail_demo.py'}"},
                    },
                    cwd=root,
                )
                auditor.observe_tool_result({"toolCallId": "call-1", "status": "failed"})
                await asyncio.sleep(0)

        self.assertEqual(_FakeAsyncClient.calls, [])


if __name__ == "__main__":
    unittest.main()
