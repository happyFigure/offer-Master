import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class FileOperationPolicyTest(unittest.TestCase):
    def test_copy_intent_without_model_name_never_generates_destination(self) -> None:
        from app.agent_runtime.context.file_operation_policy import resolve_copy_destination_from_operation_intent

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "resume.tex"
            source.write_text("resume", encoding="utf-8")

            target = resolve_copy_destination_from_operation_intent(
                {
                    "destination": {"kind": "directory", "path": temp_dir},
                    "name_policy": "copy_suffix",
                    "user_delegated_name": True,
                    "avoid_conflict": True,
                },
                str(source),
            )

        self.assertEqual("", target)

    def test_structured_copy_intent_rejects_destination_kind_mismatch(self) -> None:
        from app.agent_runtime.context.file_operation_policy import resolve_copy_destination_from_operation_intent

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "resume.tex"
            existing_file = root / "existing.txt"
            existing_directory = root / "existing-dir"
            source.write_text("resume", encoding="utf-8")
            existing_file.write_text("file", encoding="utf-8")
            existing_directory.mkdir()

            self.assertEqual(
                "",
                resolve_copy_destination_from_operation_intent(
                    {
                        "destination": {"kind": "directory", "path": str(existing_file)},
                        "name_policy": "copy_suffix",
                        "user_delegated_name": True,
                    },
                    str(source),
                ),
            )
            self.assertEqual(
                "",
                resolve_copy_destination_from_operation_intent(
                    {"destination": {"kind": "file", "path": str(existing_directory)}},
                    str(source),
                ),
            )

    def test_structured_copy_content_based_policy_does_not_invent_generic_name(self) -> None:
        from app.agent_runtime.context.file_operation_policy import resolve_copy_destination_from_operation_intent

        source = r"C:\Users\phoenix\Documents\简历\一下.tex"

        self.assertEqual(
            "",
            resolve_copy_destination_from_operation_intent(
                {
                    "destination": {"kind": "directory", "path": r"C:\Users\phoenix\Documents\简历"},
                    "name_policy": "content_based",
                    "user_delegated_name": True,
                },
                source,
                path_exists=lambda _path: False,
            ),
        )

    def test_structured_copy_content_based_policy_uses_model_name_intent(self) -> None:
        from app.agent_runtime.context.file_operation_policy import resolve_copy_destination_from_operation_intent

        source = r"C:\Users\phoenix\Documents\简历\一下.tex"
        target_directory = r"C:\Users\phoenix\Documents\简历"

        self.assertEqual(
            r"C:\Users\phoenix\Documents\简历\刘汉卿_AI_Agent_Resume.tex",
            resolve_copy_destination_from_operation_intent(
                {
                    "destination": {"kind": "directory", "path": target_directory},
                    "name_policy": "content_based",
                    "user_delegated_name": True,
                    "name_intent": {
                        "filename_stem": "刘汉卿_AI_Agent_Resume",
                        "source_basis": "file_content",
                    },
                },
                source,
                path_exists=lambda _path: False,
            ),
        )

    def test_structured_rename_intent_resolves_model_selected_file(self) -> None:
        from app.agent_runtime.context.file_operation_policy import generate_rename_destination_from_operation_intent

        source = r"C:\Users\phoenix\Documents\简历\一下.tex"
        target = r"C:\Users\phoenix\Documents\简历\刘汉卿-AI-Agent-后端简历.tex"

        self.assertEqual(
            target,
            generate_rename_destination_from_operation_intent(
                {
                    "destination": {"kind": "file", "path": target},
                    "name_policy": "content_based",
                    "name_intent": {"filename": target.rsplit("\\", 1)[-1], "source_basis": "file_content"},
                },
                source,
                path_exists=lambda _path: False,
            ),
        )

    def test_structured_rename_name_intent_normalizes_display_name_and_extension(self) -> None:
        from app.agent_runtime.context.file_operation_policy import generate_rename_destination_from_operation_intent

        source = r"C:\Users\phoenix\Documents\简历\一下.tex"
        result = generate_rename_destination_from_operation_intent(
            {
                "destination": {"kind": "directory", "path": r"C:\Users\phoenix\Documents\简历"},
                "name_intent": {
                    "display_name": "Agent-First 工程师简历：AI 智能体平台研发 × 高可靠后端实践",
                    "source_basis": "file_content",
                },
            },
            source,
            path_exists=lambda _path: False,
        )

        self.assertEqual(
            r"C:\Users\phoenix\Documents\简历\Agent-First_工程师简历_AI_智能体平台研发_高可靠后端实践.tex",
            result,
        )

    def test_structured_rename_rejects_placeholder_name(self) -> None:
        from app.agent_runtime.context.file_operation_policy import generate_rename_destination_from_operation_intent

        self.assertEqual(
            "",
            generate_rename_destination_from_operation_intent(
                {
                    "destination": {"kind": "file", "path": r"C:\简历\你起的名字.tex"},
                    "name_intent": {"filename": "你起的名字.tex", "source_basis": "conversation"},
                },
                r"C:\简历\一下.tex",
            ),
        )


if __name__ == "__main__":
    unittest.main()
