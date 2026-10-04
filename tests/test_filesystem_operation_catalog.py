from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class FilesystemOperationCatalogTest(unittest.TestCase):
    def test_catalog_declares_all_current_filesystem_skill_operations(self) -> None:
        from app.agent_runtime.skills.filesystem_operation_catalog import (
            get_filesystem_operation_spec,
            known_filesystem_operations,
        )

        self.assertEqual(
            ("copy_file", "path_exists", "read_file", "rename_file", "replace_text"),
            known_filesystem_operations(),
        )
        copy_spec = get_filesystem_operation_spec("copy_file")
        self.assertIsNotNone(copy_spec)
        self.assertEqual("copy_file.py", copy_spec.script_name)
        self.assertEqual(("src", "dst"), copy_spec.required_args)
        self.assertEqual("file_to_file", copy_spec.goal_kind)
        self.assertTrue(copy_spec.requires_confirmation)

    def test_executor_capability_operation_enum_comes_from_catalog(self) -> None:
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor
        from app.agent_runtime.skills.filesystem_operation_catalog import known_filesystem_operations

        schema = FilesystemSkillExecutor().capabilities()[0].input_schema

        self.assertEqual(list(known_filesystem_operations()), schema["properties"]["operation"]["enum"])

    def test_executor_capability_exposes_structured_copy_operation_intent(self) -> None:
        from app.agent_runtime.skills.filesystem_executor import FilesystemSkillExecutor

        schema = FilesystemSkillExecutor().capabilities()[0].input_schema

        self.assertIn("operation_intent", schema["properties"])
        self.assertEqual(
            ["file", "directory"],
            schema["properties"]["operation_intent"]["properties"]["destination"]["properties"]["kind"]["enum"],
        )

    def test_rename_operation_accepts_structured_model_intent(self) -> None:
        from app.agent_runtime.skills.filesystem_operation_catalog import get_filesystem_operation_spec

        spec = get_filesystem_operation_spec("rename_file")

        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertIn("operation_intent", spec.input_fields)

    def test_catalog_builds_generic_approval_payloads_for_file_to_file_operations(self) -> None:
        from app.agent_runtime.skills.filesystem_operation_catalog import (
            build_operation_approval_payload,
            format_operation_approval_summary,
            get_filesystem_operation_spec,
        )

        spec = get_filesystem_operation_spec("copy_file")
        payload = build_operation_approval_payload(spec, {"src": "A.tex", "dst": "B.tex", "overwrite": False})

        self.assertEqual("copy_file", payload["operation"])
        self.assertEqual("A.tex", payload["src"])
        self.assertEqual("B.tex", payload["dst"])
        self.assertEqual("high", payload["risk_level"])
        self.assertEqual("需要确认复制文件：A.tex -> B.tex。", format_operation_approval_summary(spec, payload))

    def test_routing_uses_catalog_for_file_to_file_arguments(self) -> None:
        from app.agent_runtime.routing.capability_routing_middleware import _filesystem_skill_tool_input
        from app.agent_runtime.skills.filesystem_operation_catalog import FILESYSTEM_OPERATION_CATALOG, FilesystemOperationSpec

        FILESYSTEM_OPERATION_CATALOG["archive_file"] = FilesystemOperationSpec(
            operation="archive_file",
            legacy_tool_name="filesystem.archive_file",
            script_name="archive_file.py",
            goal_kind="file_to_file",
            target_type="file_archive",
            required_args=("source", "target"),
            input_fields=("source", "target"),
            success_criteria=("operation_is_archive_file", "tool_result_ok", "target_path_matches"),
            result_path_arg="target",
            target_match_key="target_path",
            target_match_mode="path",
        )
        try:
            operation_intent = {
                "destination": {"kind": "file", "path": "C:/resume/archive/source.tex"},
                "name_policy": "copy_suffix",
            }
            tool_input = _filesystem_skill_tool_input(
                "归档这个文件",
                {"filesystem_operation": "archive_file", "operation_intent": operation_intent},
                {"active_file": {"path": "C:/resume/source.tex"}},
            )
        finally:
            FILESYSTEM_OPERATION_CATALOG.pop("archive_file", None)

        self.assertEqual({"user_task": "归档这个文件"}, tool_input)
        self.assertNotIn("file_task_frame", tool_input)

    def test_goal_validator_uses_catalog_for_file_to_file_success(self) -> None:
        from app.agent_runtime.goal.schemas import GoalState
        from app.agent_runtime.goal.validator import validate_goal_completion
        from app.agent_runtime.skills.filesystem_operation_catalog import FILESYSTEM_OPERATION_CATALOG, FilesystemOperationSpec

        FILESYSTEM_OPERATION_CATALOG["archive_file"] = FilesystemOperationSpec(
            operation="archive_file",
            legacy_tool_name="filesystem.archive_file",
            script_name="archive_file.py",
            goal_kind="file_to_file",
            target_type="file_archive",
            required_args=("source", "target"),
            input_fields=("source", "target"),
            success_criteria=("operation_is_archive_file", "tool_result_ok", "target_path_matches"),
            summary="归档本地文件。",
            result_path_arg="target",
            target_match_key="target_path",
            target_match_mode="path",
        )
        try:
            validation = validate_goal_completion(
                goal_state=GoalState(
                    intent="filesystem_operation",
                    target_type="file_archive",
                    expected_operation="archive_file",
                    target={"target_path": "C:/resume/archive/source.tex"},
                    success_criteria=("operation_is_archive_file", "tool_result_ok", "target_path_matches"),
                ),
                tool_name="skill.filesystem",
                tool_input={},
                result_payload={
                    "ok": True,
                    "operation": "archive_file",
                    "arguments": {"source": "C:/resume/source.tex", "target": "C:/resume/archive/source.tex"},
                },
            )
        finally:
            FILESYSTEM_OPERATION_CATALOG.pop("archive_file", None)

        self.assertTrue(validation.completed)
        self.assertEqual("final_answer", validation.next_action)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
