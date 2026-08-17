import json
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class XiaohongshuMcpScriptTest(unittest.TestCase):
    def test_xiaohongshu_scripts_print_runtime_environment_under_project_root(self):
        scripts = [
            PROJECT_ROOT / "scripts" / "login-xiaohongshu-mcp.ps1",
            PROJECT_ROOT / "scripts" / "start-xiaohongshu-mcp.ps1",
        ]

        for script in scripts:
            with self.subTest(script=script.name):
                self._assert_script_runtime_environment(script)

    def _assert_script_runtime_environment(self, script: Path):

        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                "-PrintEnvironment",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(completed.stdout)
        runtime_root = Path(payload["runtime_root"])
        self.assertTrue(runtime_root.is_relative_to(PROJECT_ROOT))
        for key in ["COOKIES_PATH", "LOCALAPPDATA", "APPDATA", "TEMP", "TMP", "HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME"]:
            value = Path(payload["env"][key])
            self.assertTrue(value.is_relative_to(PROJECT_ROOT), key)
            self.assertNotEqual("c:", value.drive.lower(), key)


if __name__ == "__main__":
    unittest.main()
