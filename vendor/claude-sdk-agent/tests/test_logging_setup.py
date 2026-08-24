from __future__ import annotations

import logging
import logging.handlers
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.logging_setup import _select_log_path, configure_logging


class LoggingSetupTests(unittest.TestCase):
    def test_select_log_path_falls_back_when_project_log_dir_is_not_writable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_log = root / "log"
            fallback_log = Path(tempfile.gettempdir()) / "claude-sdk-agent" / "test-user"
            with patch("src.logging_setup.os.access", side_effect=lambda path, mode: str(path) != str(project_log)):
                with patch.dict("src.logging_setup.os.environ", {"USER": "test-user"}, clear=False):
                    selected = _select_log_path(root)
            self.assertEqual(selected, fallback_log / "claude-sdk-agent.log")

    def test_configure_logging_keeps_stream_logging_when_file_handler_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root_logger = logging.getLogger()
            old_handlers = list(root_logger.handlers)
            old_level = root_logger.level
            try:
                for handler in list(root_logger.handlers):
                    root_logger.removeHandler(handler)
                with patch.object(logging.handlers.RotatingFileHandler, "__init__", side_effect=PermissionError("denied")):
                    path = configure_logging(root)
                self.assertTrue(str(path).endswith("claude-sdk-agent.log"))
                self.assertTrue(any(isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers))
                self.assertFalse(any(isinstance(handler, logging.handlers.RotatingFileHandler) for handler in root_logger.handlers))
            finally:
                for handler in list(root_logger.handlers):
                    root_logger.removeHandler(handler)
                for handler in old_handlers:
                    root_logger.addHandler(handler)
                root_logger.setLevel(old_level)
