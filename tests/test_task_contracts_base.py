import sys
import unittest
from pathlib import Path

from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class TaskContractsBaseTest(unittest.TestCase):
    def test_task_envelope_base_requires_shared_execution_boundary_fields(self) -> None:
        from app.agent_runtime.contracts.base import TaskEnvelopeBase

        envelope = TaskEnvelopeBase(
            schema_version="offer_master.test_task.v1",
            task_id="task-1",
            trace_id="trace-1",
            capability="test.capability",
            task_type="test.task",
            risk_level="medium",
            allowed_actions=["read_visible_page"],
            forbidden_actions=["submit_application"],
            human_approval_required=["final_submit"],
            context_refs={"job_id": "job-1"},
            metadata={"stop_before_submit": True},
        )

        self.assertEqual("test.capability", envelope.capability)
        self.assertEqual(["submit_application"], envelope.forbidden_actions)
        self.assertEqual({"job_id": "job-1"}, envelope.context_refs)

    def test_artifact_ref_rejects_extra_fields(self) -> None:
        from app.agent_runtime.contracts.base import ArtifactRef

        with self.assertRaises(ValidationError):
            ArtifactRef(type="url", title="apply", url="https://example.com", unexpected=True)

    def test_execution_result_base_requires_user_action_reason(self) -> None:
        from app.agent_runtime.contracts.base import ExecutionResultBase

        with self.assertRaisesRegex(ValidationError, "requires_user_action results require"):
            ExecutionResultBase(
                schema_version="offer_master.test_result.v1",
                task_id="task-1",
                status="waiting_user",
                requires_user_action=True,
            )


if __name__ == "__main__":
    unittest.main()
