from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.workspace_runtime import inspect_workspace_runtime, merge_observed_runtime


class WorkspaceRuntimeTests(unittest.TestCase):
    def test_inspection_separates_agent_policy_primary_resources_and_additional_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            additional = root / "additional"
            agent_config = root / "agent-config"
            skill_mount = root / "skill-mount"
            workflow_mount = agent_config / "workflows"
            for path in (workspace, additional, agent_config, skill_mount, workflow_mount):
                path.mkdir(parents=True, exist_ok=True)

            (workspace / "CLAUDE.md").write_text("project memory", encoding="utf-8")
            (workspace / "AGENTS.md").write_text("external instructions", encoding="utf-8")
            (workspace / ".claude" / "skills" / "review").mkdir(parents=True)
            (workspace / ".claude" / "skills" / "review" / "SKILL.md").write_text(
                "---\nname: review\ndescription: Review files\n---\nbody",
                encoding="utf-8",
            )
            (workspace / ".claude" / "commands").mkdir(parents=True)
            (workspace / ".claude" / "commands" / "review.md").write_text(
                "---\ndescription: Legacy review command\nargument-hint: '[path]'\n---\n$ARGUMENTS",
                encoding="utf-8",
            )
            (workspace / ".claude" / "commands" / "_shared").mkdir()
            (workspace / ".claude" / "commands" / "_shared" / "README.md").write_text(
                "# Shared command fragments",
                encoding="utf-8",
            )
            (workspace / ".claude" / "output-styles").mkdir()
            (workspace / ".claude" / "output-styles" / "concise.md").write_text(
                "---\nname: concise\ndescription: Concise output\n---\n",
                encoding="utf-8",
            )
            (workspace / ".claude" / "skills" / "internal").mkdir()
            (workspace / ".claude" / "skills" / "internal" / "SKILL.md").write_text(
                "---\nname: internal\nuser-invocable: false\n---\nbody",
                encoding="utf-8",
            )
            (workspace / ".claude" / "agents").mkdir(parents=True)
            (workspace / ".claude" / "agents" / "worker.md").write_text(
                "---\nname: worker\ndescription: Worker\n---\nPrompt",
                encoding="utf-8",
            )
            (workspace / ".claude" / "workflows").mkdir(parents=True)
            (workspace / ".claude" / "workflows" / "native.js").write_text(
                "export const meta = { name: 'native', description: 'demo', phases: [] };",
                encoding="utf-8",
            )
            (workspace / ".claude" / "settings.json").write_text(
                json.dumps(
                    {
                        "permissions": {"allow": ["Read"], "deny": ["Bash(rm *)"]},
                        "hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "secret-hook"}]}]},
                        "enableAllProjectMcpServers": False,
                        "enabledMcpjsonServers": ["shared"],
                        "env": {"PRIVATE_TOKEN": "workspace-super-secret"},
                    }
                ),
                encoding="utf-8",
            )
            (workspace / ".mcp.json").write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "shared": {
                                "command": "demo",
                                "env": {"API_KEY": "mcp-super-secret"},
                            },
                            "workspace-only": {"url": "https://example.invalid/mcp"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (workspace / ".flow" / "workflows").mkdir(parents=True)
            (workspace / ".flow" / "workflows" / "legacy.workflow.json").write_text("{}", encoding="utf-8")

            (additional / "CLAUDE.md").write_text("additional memory", encoding="utf-8")
            (additional / ".claude").mkdir()
            (additional / ".claude" / "settings.json").write_text(
                json.dumps({"hooks": {"Stop": [{"hooks": []}]}}),
                encoding="utf-8",
            )
            (additional / ".mcp.json").write_text(
                json.dumps({"mcpServers": {"additional-mcp": {"command": "ignored"}}}),
                encoding="utf-8",
            )

            (skill_mount / ".claude" / "skills" / "global-review").mkdir(parents=True)
            (skill_mount / ".claude" / "skills" / "global-review" / "SKILL.md").write_text(
                "---\nname: global-review\ndescription: Agent skill\n---\n",
                encoding="utf-8",
            )
            (workflow_mount / "global.js").write_text(
                "export const meta = { name: 'global', description: 'demo', phases: [] };",
                encoding="utf-8",
            )

            snapshot = inspect_workspace_runtime(
                cwd=workspace,
                add_dirs=[additional],
                source="agent",
                configured=True,
                setting_sources=["user", "project", "local"],
                strict_mcp_config=False,
                permission_profile="edit",
                permission_mode="acceptEdits",
                allowed_tools=["Read"],
                disallowed_tools=["Bash(rm -rf *)"],
                agent_config_dir=agent_config,
                skill_mount_root=skill_mount,
                workflow_mount_root=workflow_mount,
                agent_mcp_names=["shared", "agent-only"],
            )

            self.assertEqual(snapshot["workspace"]["source"], "agent")
            self.assertTrue(snapshot["workspace"]["configured"])
            self.assertEqual(snapshot["workspace"]["roots"][1]["nativeScope"], "access")
            self.assertEqual(snapshot["agentPolicy"]["additionalDirectories"]["mode"], "access_only")
            self.assertEqual(snapshot["effectiveRuntime"]["mcp"]["nameCollisions"], ["shared"])
            self.assertEqual(snapshot["effectiveRuntime"]["mcp"]["workspaceServerSelection"], "explicit")
            self.assertEqual(snapshot["effectiveRuntime"]["mcp"]["workspaceEnabledServerNames"], ["shared"])
            self.assertEqual(snapshot["effectiveRuntime"]["mcp"]["workspaceDisabledServerNames"], ["workspace-only"])
            self.assertNotIn("additional-mcp", snapshot["effectiveRuntime"]["mcp"]["expectedServerNames"])
            self.assertTrue(snapshot["effectiveRuntime"]["hooks"]["active"])
            self.assertEqual(snapshot["effectiveRuntime"]["permission"]["workspaceRuleCounts"]["deny"], 1)

            commands = snapshot["commands"]["items"]
            review_entries = [item for item in commands if item["name"] == "review"]
            self.assertEqual(len(review_entries), 2)
            self.assertTrue(next(item for item in review_entries if item["kind"] == "skill")["selected"])
            self.assertEqual(
                next(item for item in review_entries if item["kind"] == "skill")["invoke"]["kind"],
                "skill",
            )
            self.assertEqual(
                next(item for item in review_entries if item["kind"] == "command")["argumentHint"],
                "[path]",
            )
            support_entry = next(item for item in commands if item["name"] == "README")
            self.assertFalse(support_entry["active"])
            self.assertEqual(support_entry["inactiveReason"], "support_file")
            internal_entry = next(item for item in commands if item["name"] == "internal")
            self.assertFalse(internal_entry["active"])
            self.assertEqual(internal_entry["inactiveReason"], "not_user_invocable")
            self.assertTrue(any(item["name"] == "global-review" for item in commands))
            native_workflow = next(item for item in commands if item["name"] == "native")
            self.assertEqual(native_workflow["kind"], "workflow")
            self.assertTrue(native_workflow["invokable"])
            self.assertFalse(native_workflow["requiresConfirmation"])
            self.assertEqual(native_workflow["confirmationMode"], "none")
            self.assertEqual(native_workflow["invoke"]["kind"], "workflow")
            self.assertFalse(native_workflow["invoke"]["requiresConfirmation"])
            self.assertTrue(any(item["name"] == "global" for item in snapshot["resources"]["workflows"]["items"]))
            self.assertEqual(snapshot["resources"]["outputStyles"]["activeCount"], 1)
            self.assertTrue(any(item["name"] == "concise" for item in snapshot["resources"]["outputStyles"]["items"]))

            serialized = json.dumps(snapshot, ensure_ascii=False)
            self.assertNotIn("legacy.workflow", serialized)
            self.assertNotIn("adapter_required", serialized)
            self.assertNotIn("compatibility", snapshot["resources"])
            self.assertNotIn("workspace-super-secret", serialized)
            self.assertNotIn("mcp-super-secret", serialized)
            self.assertNotIn("secret-hook", serialized)

    def test_strict_mcp_and_setting_sources_disable_workspace_declarations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / ".claude").mkdir()
            (workspace / ".claude" / "settings.local.json").write_text(
                json.dumps({"hooks": {"Stop": [{"hooks": []}]}}),
                encoding="utf-8",
            )
            (workspace / ".mcp.json").write_text(
                json.dumps({"mcpServers": {"project": {"command": "demo"}}}),
                encoding="utf-8",
            )

            snapshot = inspect_workspace_runtime(
                cwd=workspace,
                setting_sources=["user"],
                strict_mcp_config=True,
                agent_mcp_names=["agent"],
            )

            mcp = snapshot["effectiveRuntime"]["mcp"]
            self.assertFalse(mcp["workspaceServersActive"])
            self.assertEqual(mcp["workspaceInactiveReason"], "strict_mcp_config")
            self.assertEqual(mcp["expectedServerNames"], ["agent"])
            self.assertFalse(snapshot["effectiveRuntime"]["hooks"]["active"])

    def test_permission_runtime_keeps_agent_identity_and_native_rule_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            user_config = root / "agent-config" / ".claude"
            (workspace / ".claude").mkdir(parents=True)
            user_config.mkdir(parents=True)
            (workspace / ".claude" / "settings.local.json").write_text(
                json.dumps({"permissions": {"allow": ["Read", "Bash(git status)"]}}),
                encoding="utf-8",
            )
            (user_config / "settings.json").write_text(
                json.dumps(
                    {
                        "permissions": {
                            "defaultMode": "default",
                            "allow": [f"Tool{index}" for index in range(9)],
                            "deny": ["Bash(rm -rf *)", "Read(.env)"],
                        }
                    }
                ),
                encoding="utf-8",
            )

            snapshot = inspect_workspace_runtime(
                cwd=workspace,
                setting_sources=["user", "project", "local"],
                permission_profile="bypass",
                permission_mode="bypassPermissions",
                permission_runtime_key="agent-a",
                permission_revision=7,
                agent_config_dir=user_config,
            )

            permission = snapshot["effectiveRuntime"]["permission"]
            self.assertEqual(permission["runtimeKey"], "agent-a")
            self.assertEqual(permission["revision"], 7)
            self.assertEqual(permission["permissionModeSource"], "agent_runtime")
            self.assertEqual(permission["workspaceRuleCounts"], {"allow": 2, "ask": 0, "deny": 0})
            self.assertEqual(permission["userRuleCounts"], {"allow": 9, "ask": 0, "deny": 2})
            self.assertEqual(permission["settingsRuleCounts"], {"allow": 11, "ask": 0, "deny": 2})
            self.assertEqual(permission["rulePriority"], ["deny", "ask", "allow"])
            self.assertTrue(permission["rules"][0]["effect"] == "deny")
            self.assertFalse(any(item["type"] == "workspace_permission_overridden" for item in snapshot["conflicts"]))

            full_snapshot = inspect_workspace_runtime(
                cwd=workspace,
                setting_sources=["user", "project", "local"],
                permission_profile="fullBypass",
                permission_mode="bypassPermissions",
                permission_full_bypass=True,
                permission_runtime_key="agent-b",
                agent_config_dir=user_config,
            )
            full = full_snapshot["effectiveRuntime"]["permission"]["fullBypass"]
            self.assertTrue(full["requested"])
            self.assertEqual(full["enforcementStatus"], "workspace_policy_adapter_required")
            self.assertIn("deny_rules", full["limitations"])
            self.assertFalse(full["workspaceHooksIgnored"])
            self.assertFalse(full["workspaceRulesIgnored"])

    def test_observed_runtime_redacts_sensitive_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = inspect_workspace_runtime(cwd=Path(tmp))
            observed = merge_observed_runtime(
                snapshot,
                server_info={
                    "commands": [{"name": "review", "description": "Review"}],
                    "account": {"token": "account-secret"},
                },
                mcp_status={
                    "mcpServers": [
                        {
                            "name": "demo",
                            "status": "connected",
                            "env": {"TOKEN": "mcp-secret"},
                        }
                    ]
                },
            )

            self.assertTrue(observed["observed"]["available"])
            self.assertEqual(observed["observed"]["commands"][0]["name"], "review")
            serialized = json.dumps(observed, ensure_ascii=False)
            self.assertNotIn("account-secret", serialized)
            self.assertNotIn("mcp-secret", serialized)

    def test_platform_skill_isolation_keeps_project_skills_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            skill_mount = root / "skill-mount"
            project_skill = workspace / ".claude" / "skills" / "project-review"
            platform_skill = skill_mount / ".claude" / "skills" / "global-review"
            project_skill.mkdir(parents=True)
            platform_skill.mkdir(parents=True)
            (project_skill / "SKILL.md").write_text(
                "---\nname: project-review\ndescription: Project\n---\n",
                encoding="utf-8",
            )
            (platform_skill / "SKILL.md").write_text(
                "---\nname: global-review\ndescription: Platform\n---\n",
                encoding="utf-8",
            )

            snapshot = inspect_workspace_runtime(
                cwd=workspace,
                skill_mount_root=skill_mount,
                skill_platform_catalog="exclude",
            )

            skills = snapshot["resources"]["skills"]
            project_item = next(item for item in skills["items"] if item["name"] == "project-review")
            platform_item = next(item for item in skills["items"] if item["name"] == "global-review")
            self.assertTrue(project_item["active"])
            self.assertFalse(platform_item["active"])
            self.assertEqual(platform_item["status"], "excluded")
            self.assertEqual(platform_item["reasonCodes"], ["platform_catalog_excluded"])
            self.assertEqual(skills["excludedCount"], 1)
            self.assertFalse(snapshot["agentPolicy"]["skills"]["platformMountEnabled"])
            global_command = next(
                item for item in snapshot["commands"]["items"] if item["name"] == "global-review"
            )
            self.assertFalse(global_command["active"])


if __name__ == "__main__":
    unittest.main()
