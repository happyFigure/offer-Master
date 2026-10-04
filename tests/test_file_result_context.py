import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class FileResultContextTest(unittest.TestCase):
    def test_copy_result_promotes_target_file_as_active_context(self) -> None:
        from app.agent_runtime.context.file_result_context import promote_filesystem_result_context

        metadata = {
            "active_file": {"path": r"C:\Users\phoenix\resume.tex", "last_focus": "file"},
            "recent_file_paths": [r"C:\Users\phoenix\resume.tex"],
        }
        result_payload = {
            "ok": True,
            "operation": "copy_file",
            "arguments": {
                "src": r"C:\Users\phoenix\resume.tex",
                "dst": r"C:\Users\phoenix\resume-1.tex",
            },
            "filesystem_trace": {
                "operation": "copy_file",
                "postcheck": {
                    "completed": True,
                    "source_path": r"C:\Users\phoenix\resume.tex",
                    "target_path": r"C:\Users\phoenix\resume-1.tex",
                },
            },
        }

        promoted = promote_filesystem_result_context(metadata, result_payload)

        self.assertEqual(r"C:\Users\phoenix\resume-1.tex", promoted["active_file"]["path"])
        self.assertEqual("copy_file", promoted["active_file"]["last_action"])
        self.assertEqual("filesystem_result", promoted["active_file"]["source"])
        self.assertEqual(r"C:\Users\phoenix\resume-1.tex", promoted["last_file_operation_result"]["focus_path"])
        self.assertEqual(r"C:\Users\phoenix\resume.tex", promoted["last_file_operation_result"]["source_path"])
        self.assertEqual(r"C:\Users\phoenix\resume-1.tex", promoted["last_file_operation_result"]["target_path"])
        self.assertIn(r"C:\Users\phoenix\resume.tex", promoted["recent_file_paths"])
        self.assertIn(r"C:\Users\phoenix\resume-1.tex", promoted["recent_file_paths"])

    def test_copy_result_records_generic_resource_effect_and_active_artifact(self) -> None:
        from app.agent_runtime.context.file_result_context import promote_filesystem_result_context

        promoted = promote_filesystem_result_context(
            {},
            {
                "ok": True,
                "operation": "copy_file",
                "arguments": {"src": "C:/简历/resume.tex", "dst": "C:/简历/resume-1.tex"},
                "filesystem_trace": {
                    "operation": "copy_file",
                    "postcheck": {
                        "completed": True,
                        "source_path": "C:/简历/resume.tex",
                        "target_path": "C:/简历/resume-1.tex",
                    },
                },
            },
        )

        effect = promoted["resource_effects"][-1]
        self.assertEqual("file", effect["resource_type"])
        self.assertEqual("created", effect["action"])
        self.assertEqual("copy_file", effect["operation"])
        self.assertEqual("C:/简历/resume.tex", effect["source_path"])
        self.assertEqual("C:/简历/resume-1.tex", effect["target_path"])
        self.assertEqual("C:/简历/resume-1.tex", effect["focus_path"])
        self.assertIn("刚才复制的文件", effect["aliases"])
        self.assertEqual(effect, promoted["active_resource"])
        self.assertEqual(effect, promoted["artifact_context"]["active_artifact"])


if __name__ == "__main__":
    unittest.main()
