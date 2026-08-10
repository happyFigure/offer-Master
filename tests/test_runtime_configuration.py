import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
