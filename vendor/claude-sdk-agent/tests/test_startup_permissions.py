from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.startup_permissions import repair_tree_permissions


class StartupPermissionsTests(unittest.TestCase):
    def test_repair_tree_permissions_updates_regular_paths_and_skips_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child_dir = root / "child"
            child_dir.mkdir()
            child_file = child_dir / "a.txt"
            child_file.write_text("x", encoding="utf-8")
            link = root / "link"
            link.symlink_to(child_file)

            with patch("src.startup_permissions.os.chown") as chown, patch(
                "src.startup_permissions.os.chmod"
            ) as chmod:
                changed, skipped = repair_tree_permissions(root, uid=1001, gid=1002)

            self.assertEqual(changed, 3)
            self.assertEqual(skipped, 1)
            self.assertEqual(chown.call_count, 3)
            self.assertEqual(chmod.call_count, 3)
