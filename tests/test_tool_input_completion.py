import sys
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

        self.assertEqual("C:/Users/phoenix/Documents/Obsidian Vault/简历/resume.tex", result.tool_input["path"])
        self.assertEqual(("path",), result.filled_fields)
        self.assertEqual((), result.missing_required_fields)

    def test_completes_exact_resume_name_replacement_from_recent_context(self) -> None:
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

        self.assertEqual("C:/Users/phoenix/Documents/Obsidian Vault/简历/刘汉卿-后端开发-AI-Agent平台简历.tex", result.tool_input["path"])
        self.assertEqual("刘汉卿", result.tool_input["old_text"])
        self.assertEqual("王爷", result.tool_input["new_text"])
        self.assertEqual((), result.missing_required_fields)

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
        self.assertIn("query", result.filled_fields)

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
