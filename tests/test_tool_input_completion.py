import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class ToolInputCompletionTest(unittest.TestCase):
    def test_completes_missing_filesystem_path_from_recent_context(self) -> None:
        from app.agent_runtime.tool_input_completion import complete_tool_input

        result = complete_tool_input(
            tool_name="filesystem.read_file",
            tool_input={"encoding": "utf-8"},
            input_schema={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}, "encoding": {"type": "string"}},
                "additionalProperties": False,
            },
            user_message="读取内容",
            recent_user_context="这是简历路径：C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex",
        )

        self.assertEqual(
            "C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex",
            result.tool_input["path"],
        )
        self.assertEqual(("path",), result.filled_fields)
        self.assertEqual((), result.missing_required_fields)

    def test_completes_explicit_replace_pair_only_from_user_text(self) -> None:
        from app.agent_runtime.tool_input_completion import complete_tool_input

        result = complete_tool_input(
            tool_name="filesystem.replace_text",
            tool_input={},
            input_schema={
                "type": "object",
                "required": ["path", "old_text", "new_text"],
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "additionalProperties": False,
            },
            user_message="把简历名字改为王爷，其他不要动",
            recent_user_context="这是简历路径：C:/Users/phoenix/Documents/Obsidian Vault/简历/刘汉卿-后端开发-AI-Agent平台简历.tex",
        )

        self.assertEqual(
            "C:/Users/phoenix/Documents/Obsidian Vault/简历/刘汉卿-后端开发-AI-Agent平台简历.tex",
            result.tool_input["path"],
        )
        self.assertEqual("刘汉卿", result.tool_input["old_text"])
        self.assertEqual("王爷", result.tool_input["new_text"])
        self.assertEqual((), result.missing_required_fields)

    def test_structured_rename_intent_generates_concrete_destination(self) -> None:
        from app.agent_runtime.tool_input_completion import complete_tool_input

        source_path = r"C:\简历\一下.tex"
        result = complete_tool_input(
            tool_name="filesystem.move_file",
            tool_input={
                "src": source_path,
                "operation_intent": {
                    "destination": {"kind": "directory", "path": r"C:\简历"},
                    "name_intent": {
                        "mode": "model_proposed",
                        "filename": "LiuHanqing_AI-Agent-Java-Engineer_Resume_CN.tex",
                        "source_basis": "file_content",
                        "extension_policy": "explicit",
                    },
                },
            },
            input_schema={
                "type": "object",
                "required": ["src", "dst"],
                "properties": {
                    "src": {"type": "string"},
                    "dst": {"type": "string"},
                    "overwrite": {"type": "boolean", "default": False},
                    "operation_intent": {"type": "object"},
                },
                "additionalProperties": False,
            },
            user_message="把文件名改成模型刚刚确定的名字",
            context={"active_file": {"path": source_path}},
        )

        self.assertEqual(
            r"C:\简历\LiuHanqing_AI-Agent-Java-Engineer_Resume_CN.tex",
            result.tool_input["dst"],
        )
        self.assertEqual((), result.missing_required_fields)
        self.assertEqual("model_operation_intent", result.sources["dst"])

    def test_placeholder_destination_is_not_accepted_without_structured_name(self) -> None:
        from app.agent_runtime.tool_input_completion import complete_tool_input

        result = complete_tool_input(
            tool_name="filesystem.move_file",
            tool_input={"src": r"C:\简历\一下.tex", "dst": "你起的名字.tex"},
            input_schema={
                "type": "object",
                "required": ["src", "dst"],
                "properties": {"src": {"type": "string"}, "dst": {"type": "string"}},
                "additionalProperties": False,
            },
            user_message="把文件名改成你起的名字",
            context={"active_file": {"path": r"C:\简历\一下.tex"}},
        )

        self.assertIn("dst", result.missing_required_fields)
        self.assertNotEqual("你起的名字.tex", result.tool_input.get("dst"))

    def test_plain_copy_followup_does_not_guess_same_directory_destination(self) -> None:
        from app.agent_runtime.tool_input_completion import complete_tool_input

        source_path = r"C:\简历\一下.tex"
        result = complete_tool_input(
            tool_name="filesystem.copy_file",
            tool_input={},
            input_schema={
                "type": "object",
                "required": ["src", "dst"],
                "properties": {
                    "src": {"type": "string"},
                    "dst": {"type": "string"},
                    "overwrite": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
            user_message="复制到相同目录下",
            context={"active_file": {"path": source_path}},
        )

        self.assertEqual(source_path, result.tool_input["src"])
        self.assertNotIn("dst", result.tool_input)
        self.assertEqual(("dst",), result.missing_required_fields)

    def test_structured_copy_directory_intent_requires_model_name(self) -> None:
        from app.agent_runtime.tool_input_completion import complete_tool_input

        with tempfile.TemporaryDirectory(prefix="offer-master-structured-copy-") as temp_dir:
            source_path = str(Path(temp_dir) / "resume.tex")
            Path(source_path).write_text("resume", encoding="utf-8")
            result = complete_tool_input(
                tool_name="filesystem.copy_file",
                tool_input={
                    "src": source_path,
                    "operation_intent": {
                        "destination": {"kind": "directory", "path": temp_dir},
                        "name_policy": "copy_suffix",
                        "user_delegated_name": True,
                        "avoid_conflict": True,
                    },
                },
                input_schema={
                    "type": "object",
                    "required": ["src", "dst"],
                    "properties": {
                        "src": {"type": "string"},
                        "dst": {"type": "string"},
                        "overwrite": {"type": "boolean", "default": False},
                        "operation_intent": {"type": "object"},
                    },
                    "additionalProperties": False,
                },
                user_message="给我复制一下",
            )

            self.assertNotIn("dst", result.tool_input)
            self.assertEqual(("dst",), result.missing_required_fields)

    def test_rewrites_pronoun_web_search_query_with_recent_subject(self) -> None:
        from app.agent_runtime.tool_input_completion import complete_tool_input

        result = complete_tool_input(
            tool_name="external.web_search",
            tool_input={"query": "查一下它主要业务", "max_results": 5},
            input_schema={
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                "additionalProperties": False,
            },
            user_message="查一下它主要业务",
            recent_user_context="我想了解 Canonical Ltd. 这个公司",
        )

        self.assertIn("Canonical Ltd.", result.tool_input["query"])
        self.assertIn("主要业务", result.tool_input["query"])

    def test_reports_missing_required_fields_when_context_cannot_fill_them(self) -> None:
        from app.agent_runtime.tool_input_completion import complete_tool_input

        result = complete_tool_input(
            tool_name="filesystem.read_file",
            tool_input={},
            input_schema={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
            user_message="读取内容",
            recent_user_context="",
        )

        self.assertEqual({}, result.tool_input)
        self.assertEqual(("path",), result.missing_required_fields)


if __name__ == "__main__":
    unittest.main()
