from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class FilesystemNoDeadEndTest(unittest.TestCase):
    def test_unknown_operation_outcome_carries_recovery_route(self) -> None:
        """Would fail if unknown filesystem actions can still be summarized as completed."""
        from app.agent_runtime.skills.filesystem_no_dead_end import build_filesystem_no_dead_end_outcome

        outcome = build_filesystem_no_dead_end_outcome(
            operation="unknown",
            reason="无法判断应该读文件还是检查文件存在",
            candidates=(
                {"operation": "path_exists", "score": 0.61, "risk_level": "low", "reasons": ["存在性线索"]},
                {"operation": "read_file", "score": 0.54, "risk_level": "low", "reasons": ["看下线索"]},
            ),
        )

        self.assertFalse(outcome["ok"])
        self.assertTrue(outcome["recoverable"])
        self.assertEqual("ask_user", outcome["next_action"])
        self.assertEqual("path_exists", outcome["suggested_operations"][0]["operation"])
        self.assertIn("下一步", outcome["summary"])

    def test_failed_tool_outcome_preserves_suggested_next_operation(self) -> None:
        """Would fail if failed tool calls lose the recovery hint needed by the loop."""
        from app.agent_runtime.skills.filesystem_no_dead_end import build_filesystem_no_dead_end_outcome

        outcome = build_filesystem_no_dead_end_outcome(
            operation="read_file",
            reason="用户目标是检查文件是否存在，但 read_file 没有推进目标",
            suggested_operation="path_exists",
            next_action="retry_operation",
        )

        self.assertFalse(outcome["ok"])
        self.assertTrue(outcome["recoverable"])
        self.assertEqual("retry_operation", outcome["next_action"])
        self.assertEqual("path_exists", outcome["suggested_operation"])
        self.assertIn("检查文件是否存在", outcome["reason"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
