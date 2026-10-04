import shutil
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class SdkAgentSandboxTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="sdk-agent-sandbox-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_prepare_workspace_copies_inputs_into_isolated_directories(self) -> None:
        from app.agent_runtime.sdk_agents.sandbox import SdkAgentSandboxManager

        source_resume = self.tmpdir / "resume_A.md"
        source_jd = self.tmpdir / "jd.txt"
        source_resume.write_text("resume A", encoding="utf-8")
        source_jd.write_text("java backend jd", encoding="utf-8")

        manager = SdkAgentSandboxManager(base_dir=self.tmpdir / "runs")
        workspace = manager.prepare_workspace(
            run_id="run_001",
            inputs={
                "input/resume.md": source_resume,
                "input/jd.txt": source_jd,
            },
        )

        self.assertEqual("run_001", workspace.run_id)
        self.assertEqual("temp_copy", workspace.mode)
        self.assertTrue((workspace.input_dir / "resume.md").is_file())
        self.assertTrue((workspace.input_dir / "jd.txt").is_file())
        self.assertEqual("resume A", (workspace.input_dir / "resume.md").read_text(encoding="utf-8"))
        self.assertEqual("java backend jd", (workspace.input_dir / "jd.txt").read_text(encoding="utf-8"))
        self.assertTrue(workspace.work_dir.is_dir())
        self.assertTrue(workspace.output_dir.is_dir())

    def test_prepare_workspace_rejects_run_id_that_escapes_base_dir(self) -> None:
        from app.agent_runtime.sdk_agents.sandbox import SandboxPathError, SdkAgentSandboxManager

        manager = SdkAgentSandboxManager(base_dir=self.tmpdir / "runs")

        with self.assertRaises(SandboxPathError):
            manager.prepare_workspace(run_id="../escape")

    def test_resolve_path_allows_reads_inside_sandbox(self) -> None:
        from app.agent_runtime.sdk_agents.sandbox import SdkAgentSandboxManager

        manager = SdkAgentSandboxManager(base_dir=self.tmpdir / "runs")
        workspace = manager.prepare_workspace(run_id="run_001")

        resolved = manager.resolve_path(workspace, "input/resume.md", access="read")

        self.assertEqual(workspace.input_dir / "resume.md", resolved)

    def test_resolve_path_rejects_absolute_paths_and_traversal(self) -> None:
        from app.agent_runtime.sdk_agents.sandbox import SandboxPathError, SdkAgentSandboxManager

        manager = SdkAgentSandboxManager(base_dir=self.tmpdir / "runs")
        workspace = manager.prepare_workspace(run_id="run_001")

        with self.assertRaises(SandboxPathError):
            manager.resolve_path(workspace, r"C:\Users\phoenix\Documents\resume.docx", access="read")

        with self.assertRaises(SandboxPathError):
            manager.resolve_path(workspace, "../../Documents/resume.docx", access="read")

    def test_resolve_path_keeps_input_read_only(self) -> None:
        from app.agent_runtime.sdk_agents.sandbox import SandboxPathError, SdkAgentSandboxManager

        manager = SdkAgentSandboxManager(base_dir=self.tmpdir / "runs")
        workspace = manager.prepare_workspace(run_id="run_001")

        with self.assertRaises(SandboxPathError):
            manager.resolve_path(workspace, "input/resume.md", access="write")

        self.assertEqual(workspace.work_dir / "draft.md", manager.resolve_path(workspace, "work/draft.md", access="write"))
        self.assertEqual(
            workspace.output_dir / "resume_tailored.md",
            manager.resolve_path(workspace, "output/resume_tailored.md", access="write"),
        )

    def test_collect_artifacts_returns_only_output_files(self) -> None:
        from app.agent_runtime.sdk_agents.sandbox import SdkAgentSandboxManager

        manager = SdkAgentSandboxManager(base_dir=self.tmpdir / "runs")
        workspace = manager.prepare_workspace(run_id="run_001")
        (workspace.work_dir / "draft.md").write_text("draft", encoding="utf-8")
        (workspace.output_dir / "resume_tailored.md").write_text("final", encoding="utf-8")

        artifacts = manager.collect_artifacts(workspace)

        self.assertEqual(
            [
                {
                    "logical_path": "output/resume_tailored.md",
                    "size_bytes": len("final"),
                    "kind": "md",
                }
            ],
            artifacts,
        )


if __name__ == "__main__":
    unittest.main()
