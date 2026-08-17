import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class RuntimeConfigurationTest(unittest.TestCase):
    def test_local_runtime_paths_stay_inside_project_root(self):
        from app.core.config import Settings

        settings = Settings(_env_file=None)

        self.assertEqual(PROJECT_ROOT / "logs", settings.log_dir)
        self.assertEqual(PROJECT_ROOT / "data" / "uploads", settings.uploads_path)
        self.assertEqual(PROJECT_ROOT / "data" / "imports", settings.imports_path)
        self.assertEqual(PROJECT_ROOT / "data" / "exports", settings.exports_path)
        self.assertNotEqual("c:", settings.log_dir.drive.lower())

    def test_feature_defaults_match_current_mvp_boundaries(self):
        from app.core.config import Settings

        settings = Settings(_env_file=None)

        self.assertEqual(["mock", "import_file"], settings.enabled_job_providers)
        self.assertFalse(settings.mcp_enabled)
        self.assertEqual("web_speech", settings.speech_provider)
        self.assertEqual(30, settings.worker_poll_interval_seconds)
        self.assertEqual(3, settings.worker_max_retries)

    def test_xiaohongshu_rest_service_can_be_configured_without_c_drive_runtime_state(self):
        from app.core.config import Settings

        with patch.dict(
            "os.environ",
            {
                "JOBPILOT_XIAOHONGSHU_MCP_BASE_URL": "http://127.0.0.1:18060/",
                "JOBPILOT_XIAOHONGSHU_MCP_AUTH_TOKEN": "local-test-token",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        self.assertEqual("http://127.0.0.1:18060", settings.xiaohongshu_mcp_base_url)
        self.assertEqual("local-test-token", settings.xiaohongshu_mcp_auth_token.get_secret_value())
        self.assertNotEqual("c:", settings.imports_path.drive.lower())


if __name__ == "__main__":
    unittest.main()
