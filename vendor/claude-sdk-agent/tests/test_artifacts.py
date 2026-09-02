from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from src.artifacts import ArtifactStore, ClaudeArtifactService


class ClaudeArtifactServiceTest(unittest.TestCase):
    def test_save_run_from_sdk_affected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            output = workspace / "report.md"
            output.write_text("hello", encoding="utf-8")
            service = ClaudeArtifactService(
                ArtifactStore(root / "data" / "artifacts"),
                runtime_root=root,
            )

            record = asyncio.run(
                service.save_run_from_paths(
                    session_id="session-1",
                    run_id="run-1",
                    workspace_cwd=workspace,
                    workspace_add_dirs=[],
                    affected_files=["report.md", str(root / "outside.txt")],
                    started_at=1.0,
                    status="completed",
                )
            )

            self.assertEqual(record["runtime"], "claude-sdk-agent")
            self.assertEqual(record["summary"]["artifactCount"], 1)
            self.assertEqual(record["summary"]["modified"], 1)
            self.assertEqual(record["artifacts"][0]["source"], "sdk_affected_files")
            self.assertEqual(record["artifacts"][0]["relativePath"], "report.md")
            self.assertEqual(record["artifacts"][0]["availableActions"], ["open", "download"])
            self.assertIn("outside_workspace", record["errors"][0])

            stored = asyncio.run(service.get_run("run-1"))
            self.assertIsNotNone(stored)
            artifact = asyncio.run(service.get_artifact(record["artifacts"][0]["artifactId"]))
            self.assertEqual(artifact["path"], str(output.resolve()))


if __name__ == "__main__":
    unittest.main()
