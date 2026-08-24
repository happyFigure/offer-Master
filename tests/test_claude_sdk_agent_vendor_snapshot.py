from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "claude-sdk-agent"


class ClaudeSdkAgentVendorSnapshotTest(unittest.TestCase):
    def test_vendor_snapshot_keeps_source_without_runtime_state(self) -> None:
        self.assertTrue((VENDOR_ROOT / "README.md").is_file())
        self.assertTrue((VENDOR_ROOT / "src" / "main.py").is_file())
        self.assertTrue((VENDOR_ROOT / "config" / "service.json").is_file())
        self.assertTrue((VENDOR_ROOT / "tests").is_dir())

        for runtime_dir in (".venv", "venv", ".claude", "data", "log", "logs"):
            self.assertFalse((VENDOR_ROOT / runtime_dir).exists(), runtime_dir)

    def test_integration_docs_and_start_script_are_present(self) -> None:
        doc = PROJECT_ROOT / "docs" / "integrations" / "claude-sdk-agent.md"
        script = PROJECT_ROOT / "scripts" / "start_claude_sdk_agent.ps1"

        self.assertTrue(doc.is_file())
        self.assertTrue(script.is_file())
        self.assertIn("external.web_search", doc.read_text(encoding="utf-8"))
        script_text = script.read_text(encoding="utf-8")
        self.assertIn("vendor\\claude-sdk-agent", script_text)
        self.assertIn("CLAUDE_SDK_AGENT_CLI_PATH", script_text)

        runtime_text = (VENDOR_ROOT / "src" / "claude_sdk" / "runtime.py").read_text(encoding="utf-8")
        self.assertIn('"ANTHROPIC_API_KEY": proxy_token', runtime_text)
        self.assertIn('"ANTHROPIC_API_KEY",', runtime_text)
        self.assertIn("final_result_error_text", runtime_text)

        stream_adapter_text = (VENDOR_ROOT / "src" / "claude_sdk" / "stream_adapter.py").read_text(encoding="utf-8")
        self.assertIn("def final_result_error_text", stream_adapter_text)
        self.assertIn("达到最大回合数", stream_adapter_text)


if __name__ == "__main__":
    unittest.main()
