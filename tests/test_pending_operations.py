import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class PendingOperationFrameTest(unittest.TestCase):
    def test_pending_operation_round_trips_through_metadata(self) -> None:
        from app.agent_runtime.pending_operations import (
            PendingOperationFrame,
            pending_operation_from_metadata,
            with_pending_operation,
        )

        frame = PendingOperationFrame(
            capability="skill.filesystem",
            operation="copy_file",
            known_args={"src": "C:/resume.tex", "overwrite": False},
            missing_args=("dst",),
            constraints={"target_directory": "same_as_source", "name_policy": "numeric_suffix"},
            resume_hints=("用户补目标文件名或允许系统自动起名",),
        )

        metadata = with_pending_operation({}, frame)
        recovered = pending_operation_from_metadata(metadata)

        self.assertIsNotNone(recovered)
        self.assertEqual("copy_file", recovered.operation)
        self.assertEqual("C:/resume.tex", recovered.known_args["src"])
        self.assertEqual(("dst",), recovered.missing_args)
        self.assertEqual("same_as_source", recovered.constraints["target_directory"])

    def test_clear_pending_operation_keeps_unrelated_context(self) -> None:
        from app.agent_runtime.pending_operations import (
            PendingOperationFrame,
            clear_pending_operation,
            with_pending_operation,
        )

        frame = PendingOperationFrame(
            capability="skill.filesystem",
            operation="copy_file",
            known_args={"src": "C:/resume.tex"},
            missing_args=("dst",),
        )

        metadata = with_pending_operation({"active_file": {"path": "C:/resume.tex"}}, frame)
        cleared = clear_pending_operation(metadata)

        self.assertNotIn("pending_operation", cleared)
        self.assertEqual("C:/resume.tex", cleared["active_file"]["path"])

    def test_builds_pending_copy_from_missing_tool_input_completion(self) -> None:
        from app.agent_runtime.pending_operations import build_pending_operation_from_tool_input_completion
        from app.agent_runtime.tool_input_completion import ToolInputCompletionResult

        completion = ToolInputCompletionResult(
            tool_input={"src": "C:/简历/刘汉卿.tex", "overwrite": False},
            missing_required_fields=("dst",),
        )

        frame = build_pending_operation_from_tool_input_completion(
            tool_name="filesystem.copy_file",
            completion=completion,
            context_metadata={
                "filesystem_operation": "copy_file",
                "active_file": {"path": "C:/简历/刘汉卿.tex"},
            },
            user_message="给我复制一下这个文件在同一个目录下，名字带数字，不要冲突了",
        )

        self.assertIsNotNone(frame)
        self.assertEqual("skill.filesystem", frame.capability)
        self.assertEqual("copy_file", frame.operation)
        self.assertEqual("C:/简历/刘汉卿.tex", frame.known_args["src"])
        self.assertEqual(("dst",), frame.missing_args)
        self.assertEqual({}, frame.constraints)


if __name__ == "__main__":
    unittest.main()
