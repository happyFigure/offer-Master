from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.main import _drop_to_sudo_user, _repair_repo_permissions_for_sudo_user


class MainTests(unittest.TestCase):
    def test_drop_to_sudo_user_is_noop_on_windows(self) -> None:
        with patch("src.main.os.name", "nt"), patch("src.main.os.geteuid") as geteuid:
            _drop_to_sudo_user()
            geteuid.assert_not_called()

    def test_drop_to_sudo_user_switches_identity_and_env(self) -> None:
        target = SimpleNamespace(
            pw_name="demo",
            pw_uid=1001,
            pw_gid=1002,
            pw_dir="/home/demo",
            pw_shell="/bin/bash",
        )
        with patch("src.main.os.geteuid", return_value=0), patch.dict(
            "src.main.os.environ",
            {"SUDO_USER": "demo", "SUDO_UID": "1001", "SUDO_GID": "1002"},
            clear=True,
        ), patch("pwd.getpwuid", return_value=target), patch(
            "src.main.os.initgroups"
        ) as initgroups, patch("src.main.os.setgid") as setgid, patch("src.main.os.setuid") as setuid:
            _drop_to_sudo_user()
            initgroups.assert_called_once_with("demo", 1002)
            setgid.assert_called_once_with(1002)
            setuid.assert_called_once_with(1001)

    def test_repair_repo_permissions_runs_for_sudo_target(self) -> None:
        target = SimpleNamespace(
            pw_name="demo",
            pw_uid=1001,
            pw_gid=1002,
            pw_dir="/home/demo",
            pw_shell="/bin/bash",
        )
        with patch("src.main.os.name", "posix"), patch("src.main.os.geteuid", return_value=0), patch.dict(
            "src.main.os.environ",
            {"SUDO_USER": "demo", "SUDO_UID": "1001", "SUDO_GID": "1002"},
            clear=True,
        ), patch("pwd.getpwuid", return_value=target), patch(
            "src.main.repair_tree_permissions"
        ) as repair:
            repair.return_value = (12, 3)
            _repair_repo_permissions_for_sudo_user()
            repair.assert_called_once()
