import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


def sample_find_apply_entry_envelope():
    from app.agent_runtime.external_tasks.schemas import (
        ExternalTaskCandidateProfileRef,
        ExternalTaskJobContext,
        FindApplyEntryTaskEnvelope,
    )

    return FindApplyEntryTaskEnvelope(
        task_id="handoff-task-1",
        trace_id="trace-handoff-1",
        job=ExternalTaskJobContext(
            job_id="job-1",
            company_name="Tencent",
            title="Backend Engineer Intern",
            source_url="https://careers.tencent.com/job/1",
            apply_url_candidate="https://careers.tencent.com/apply/1",
        ),
        candidate_profile_ref=ExternalTaskCandidateProfileRef(
            profile_id="default",
            resume_version_id="resume-v3",
        ),
    )


class TaskHandoffAndSafetyTest(unittest.TestCase):
    def test_handoff_builder_compiles_find_apply_entry_into_browser_task(self) -> None:
        from app.agent_runtime.contracts.handoff import HandoffPayloadBuilder
        from app.agent_runtime.contracts.tasks.browser_application import BrowserTaskType
        from app.agent_runtime.tool_registry import APPLICATION_FIND_APPLY_ENTRY_TOOL

        envelope = HandoffPayloadBuilder().build_browser_application_task(
            find_apply_entry_envelope=sample_find_apply_entry_envelope()
        )

        self.assertEqual(APPLICATION_FIND_APPLY_ENTRY_TOOL, envelope.capability)
        self.assertEqual(BrowserTaskType.PREPARE_APPLICATION, envelope.task_type)
        self.assertEqual("https://careers.tencent.com/apply/1", envelope.start_url)
        self.assertEqual({"job_id": "job-1", "profile_id": "default", "resume_version_id": "resume-v3"}, envelope.context_refs)
        self.assertTrue(envelope.stop_before_submit)
        self.assertIn("submit_application", envelope.forbidden_actions)

    def test_browser_safety_gate_blocks_result_that_claims_final_submit(self) -> None:
        from app.agent_runtime.contracts.handoff import HandoffPayloadBuilder
        from app.agent_runtime.contracts.safety import BrowserSafetyGate
        from app.agent_runtime.contracts.tasks.browser_application import BrowserExecutionResult, BrowserExecutionStatus

        envelope = HandoffPayloadBuilder().build_browser_application_task(
            find_apply_entry_envelope=sample_find_apply_entry_envelope()
        )
        result = BrowserExecutionResult(
            task_id="handoff-task-1",
            status=BrowserExecutionStatus.SUBMITTED,
            summary="Submitted the application.",
            submitted=True,
            current_url="https://careers.tencent.com/apply/1/complete",
        )

        guard_result = BrowserSafetyGate().validate_result(envelope=envelope, result=result)

        self.assertFalse(guard_result.allowed)
        self.assertEqual("final_submit_forbidden", guard_result.reason_code)
        self.assertIn("final submit", guard_result.message.lower())


if __name__ == "__main__":
    unittest.main()
