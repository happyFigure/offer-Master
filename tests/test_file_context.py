import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class StructuredFileContextTest(unittest.TestCase):
    def test_plain_language_only_carries_active_file_facts(self) -> None:
        from app.agent_runtime.context.file_context import build_file_context_hints

        source_path = r"C:\简历\一下.tex"
        hints = build_file_context_hints(
            user_message="复制一份这个文件到相同目录下",
            recent_user_context=f"当前文件是 {source_path}",
        )

        self.assertEqual(source_path, hints["active_file"]["path"])
        self.assertNotIn("filesystem_operation", hints)
        self.assertNotIn("operation_intent", hints)

    def test_structured_context_is_forwarded_without_reparsing_prose(self) -> None:
        from app.agent_runtime.context.file_context import build_file_context_hints

        source_path = r"C:\简历\一下.tex"
        intent = {
            "destination": {"kind": "directory", "path": r"C:\简历"},
            "name_policy": "copy_suffix",
            "user_delegated_name": True,
            "avoid_conflict": True,
        }
        hints = build_file_context_hints(
            user_message="给我复制一下",
            context={
                "active_file": {"path": source_path},
                "filesystem_operation": "copy_file",
                "operation_intent": intent,
            },
        )

        self.assertEqual("copy_file", hints["filesystem_operation"])
        self.assertEqual(intent, hints["operation_intent"])

    def test_copy_requires_structured_destination_and_does_not_guess_same_directory_name(self) -> None:
        from app.agent_runtime.context.file_context import complete_copy_file_arguments

        source_path = r"C:\简历\一下.tex"
        result = complete_copy_file_arguments(
            tool_input={"src": source_path},
            user_message="复制到相同目录下",
            context={"active_file": {"path": source_path}},
        )

        self.assertEqual(source_path, result["src"])
        self.assertNotIn("dst", result)

    def test_copy_structured_directory_intent_requires_model_name(self) -> None:
        from app.agent_runtime.context.file_context import complete_copy_file_arguments

        with tempfile.TemporaryDirectory(prefix="offer-master-structured-copy-") as temp_dir:
            source_path = str(Path(temp_dir) / "resume.tex")
            Path(source_path).write_text("resume", encoding="utf-8")
            intent = {
                "destination": {"kind": "directory", "path": temp_dir},
                "name_policy": "copy_suffix",
                "user_delegated_name": True,
                "avoid_conflict": True,
            }
            result = complete_copy_file_arguments(
                tool_input={"src": source_path, "operation_intent": intent},
                user_message="你自己起一个合适的名字",
                context={"active_file": {"path": source_path}},
            )

            self.assertNotIn("dst", result)

    def test_rename_structured_name_intent_generates_concrete_destination(self) -> None:
        from app.agent_runtime.context.file_context import complete_move_file_arguments

        source_path = r"C:\简历\一下.tex"
        intent = {
            "destination": {"kind": "directory", "path": r"C:\简历"},
            "name_intent": {
                "display_name": "LiuHanqing_AI-Agent-Java-Engineer_Resume_CN.tex",
                "source_basis": "file_content",
            },
        }
        result = complete_move_file_arguments(
            tool_input={"src": source_path, "operation_intent": intent},
            user_message="把名字修改为你起的名字",
            context={"active_file": {"path": source_path}},
        )

        self.assertEqual(
            r"C:\简历\LiuHanqing_AI-Agent-Java-Engineer_Resume_CN.tex",
            result["dst"],
        )

    def test_plain_filename_proposal_in_context_is_ignored(self) -> None:
        from app.agent_runtime.context.file_context import build_file_context_hints

        source_path = r"C:\简历\一下.tex"
        hints = build_file_context_hints(
            user_message="现在把名字修改为你起的名字",
            context={
                "active_file": {"path": source_path},
                "file_name_proposal": {"filename": "should-not-be-used.tex"},
            },
        )

        self.assertNotIn("file_name_proposal", hints)
        self.assertNotIn("operation_intent", hints)


if __name__ == "__main__":
    unittest.main()
