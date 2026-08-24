from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import SkillSettings, WorkflowSettings
from src.skills_mount import sync_skill_mount
from src.workflows_mount import sync_workflow_mount


class SkillsMountTests(unittest.TestCase):
    def test_sync_skill_mount_links_skills_from_multiple_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src1 = root / "src1"
            src2 = root / "src2"
            (src1 / "alpha").mkdir(parents=True)
            (src1 / "alpha" / "SKILL.md").write_text("alpha", encoding="utf-8")
            (src2 / "beta").mkdir(parents=True)
            (src2 / "beta" / "SKILL.md").write_text("beta", encoding="utf-8")
            mount = root / "mount"

            mount_root, names = sync_skill_mount(
                SkillSettings(source_dirs=[src1, src2], mount_dir=mount)
            )

            self.assertEqual(mount_root, mount)
            self.assertEqual(names, ["alpha", "beta"])
            self.assertTrue((mount / ".claude" / "skills" / "alpha").is_symlink())
            self.assertTrue((mount / ".claude" / "skills" / "beta").is_symlink())

    def test_first_source_wins_on_duplicate_skill_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src1 = root / "src1"
            src2 = root / "src2"
            (src1 / "dup").mkdir(parents=True)
            (src1 / "dup" / "SKILL.md").write_text("alpha", encoding="utf-8")
            (src2 / "dup").mkdir(parents=True)
            (src2 / "dup" / "SKILL.md").write_text("beta", encoding="utf-8")
            mount = root / "mount"

            sync_skill_mount(SkillSettings(source_dirs=[src1, src2], mount_dir=mount))

            target = mount / ".claude" / "skills" / "dup"
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(), (src1 / "dup").resolve())

    def test_sync_workflow_mount_links_named_workflows_to_target_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src1 = root / "src1"
            src2 = root / "src2"
            src1.mkdir()
            src2.mkdir()
            (src1 / "review.js").write_text("export const meta = {}", encoding="utf-8")
            (src2 / "research").mkdir()
            (src2 / "research" / "index.js").write_text("export const meta = {}", encoding="utf-8")
            target_root = root / ".claude" / "workflows"

            workflow_root, names = sync_workflow_mount(
                WorkflowSettings(source_dirs=[src1, src2], target_dir=target_root)
            )

            self.assertEqual(workflow_root, target_root)
            self.assertEqual(names, ["review.js", "research"])
            self.assertTrue((target_root / "review.js").is_symlink())
            self.assertTrue((target_root / "research").is_symlink())

    def test_sync_workflow_mount_first_source_wins_on_duplicate_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src1 = root / "src1"
            src2 = root / "src2"
            src1.mkdir()
            src2.mkdir()
            (src1 / "dup.js").write_text("first", encoding="utf-8")
            (src2 / "dup.js").write_text("second", encoding="utf-8")
            target_root = root / ".claude" / "workflows"

            sync_workflow_mount(WorkflowSettings(source_dirs=[src1, src2], target_dir=target_root))

            target = target_root / "dup.js"
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(), (src1 / "dup.js").resolve())

    def test_sync_workflow_mount_preserves_local_target_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            src.mkdir()
            (src / "managed.js").write_text("managed", encoding="utf-8")
            target_root = root / ".claude" / "workflows"
            target_root.mkdir(parents=True)
            local = target_root / "local.js"
            local.write_text("local", encoding="utf-8")

            sync_workflow_mount(WorkflowSettings(source_dirs=[src], target_dir=target_root))

            self.assertEqual(local.read_text(encoding="utf-8"), "local")
            self.assertTrue((target_root / "managed.js").is_symlink())
