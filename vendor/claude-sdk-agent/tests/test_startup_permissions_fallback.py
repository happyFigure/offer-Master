from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.startup_permissions import repair_tree_permissions


class StartupPermissionsFallbackTests(unittest.TestCase):
    def test_repair_tree_permissions_falls_back_when_follow_symlinks_is_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_path = root / "a.txt"
            file_path.write_text("x", encoding="utf-8")
            chown_calls = []
            chmod_calls = []

            def fake_chown(path, uid, gid, follow_symlinks=True):
                chown_calls.append((path, uid, gid, follow_symlinks))
                if follow_symlinks is False:
                    raise NotImplementedError()

            def fake_chmod(path, mode, follow_symlinks=True):
                chmod_calls.append((path, mode, follow_symlinks))
                if follow_symlinks is False:
                    raise NotImplementedError()

            with patch("src.startup_permissions.os.chown", side_effect=fake_chown), patch(
                "src.startup_permissions.os.chmod", side_effect=fake_chmod
            ):
                changed, skipped = repair_tree_permissions(root, uid=1001, gid=1002)

            self.assertEqual(changed, 2)
            self.assertEqual(skipped, 0)
            self.assertTrue(any(call[-1] is False for call in chown_calls))
            self.assertTrue(any(call[-1] is True for call in chown_calls))
            self.assertTrue(any(call[-1] is False for call in chmod_calls))
            self.assertTrue(any(call[-1] is True for call in chmod_calls))
