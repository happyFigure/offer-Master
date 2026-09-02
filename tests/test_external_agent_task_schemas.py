import sys
import unittest
from pathlib import Path

from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class ExternalAgentTaskSchemasTest(unittest.TestCase):
    def test_find_apply_entry_envelope_defaults_to_safe_task_contract(self):
        from app.agent_runtime.external_tasks.schemas import (
            ExternalTaskCandidateProfileRef,
            ExternalTaskJobContext,
            ExternalTaskType,
            FindApplyEntryTaskEnvelope,
        )

        envelope = FindApplyEntryTaskEnvelope(
            task_id="task-001",
            trace_id="trace-001",
            job=ExternalTaskJobContext(
                job_id="job-001",
                company_name="Tencent",
                title="Java Backend Engineer - 2027 Campus",
                source_url="https://join.qq.com/campus/job/001",
                jd_summary="Campus backend role requiring Java and distributed systems.",
            ),
            candidate_profile_ref=ExternalTaskCandidateProfileRef(
                profile_id="default",
                resume_version_id="resume-v3",
            ),
        )

        self.assertEqual(ExternalTaskType.FIND_APPLY_ENTRY, envelope.task_type)
        self.assertEqual("offer_master.find_apply_entry_task.v1", envelope.schema_version)
        self.assertIn("open_browser", envelope.allowed_actions)
        self.assertIn("click_apply_button", envelope.allowed_actions)
        self.assertIn("submit_application", envelope.forbidden_actions)
        self.assertIn("answer_sensitive_questions", envelope.forbidden_actions)
        self.assertIn("final_submit", envelope.human_approval_required)
        self.assertIn("screenshot_path", envelope.evidence_required)
        self.assertEqual("ApplyEntryDiscoveryResultV1", envelope.output_schema)
        self.assertEqual("job-001", envelope.model_dump(mode="json")["job"]["job_id"])

    def test_found_opened_result_requires_apply_or_browser_url_and_evidence(self):
        from app.agent_runtime.external_tasks.schemas import (
            ApplyEntryDiscoveryResult,
            ApplyEntryDiscoveryStatus,
            ExternalAgentArtifactRef,
        )

        with self.assertRaises(ValidationError):
            ApplyEntryDiscoveryResult(
                task_id="task-001",
                status=ApplyEntryDiscoveryStatus.FOUND_OPENED,
                confidence=0.9,
            )

        result = ApplyEntryDiscoveryResult(
            task_id="task-001",
            status=ApplyEntryDiscoveryStatus.FOUND_OPENED,
            apply_url="https://join.qq.com/campus/apply/001",
            final_browser_url="https://join.qq.com/campus/apply/001/form",
            platform="company_site",
            button_text="立即投递",
            confidence=0.91,
            evidence_artifacts=[
                ExternalAgentArtifactRef(
                    artifact_type="screenshot",
                    path_or_uri="F:/System_temp/apply-entry.png",
                )
            ],
            next_action="wait_user_review",
        )

        self.assertEqual("found_opened", result.status.value)
        self.assertEqual("https://join.qq.com/campus/apply/001", result.apply_url)
        self.assertEqual("screenshot", result.evidence_artifacts[0].artifact_type)

    def test_blocked_result_requires_blocked_reason(self):
        from app.agent_runtime.external_tasks.schemas import (
            ApplyEntryBlockedReason,
            ApplyEntryDiscoveryResult,
            ApplyEntryDiscoveryStatus,
        )

        with self.assertRaises(ValidationError):
            ApplyEntryDiscoveryResult(
                task_id="task-001",
                status=ApplyEntryDiscoveryStatus.BLOCKED,
                confidence=0.4,
            )

        result = ApplyEntryDiscoveryResult(
            task_id="task-001",
            status=ApplyEntryDiscoveryStatus.BLOCKED,
            blocked_reason=ApplyEntryBlockedReason.LOGIN_REQUIRED,
            confidence=0.5,
            candidate_urls=["https://join.qq.com/login"],
            next_action="ask_user_login",
        )

        self.assertEqual(ApplyEntryBlockedReason.LOGIN_REQUIRED, result.blocked_reason)
        self.assertEqual(["https://join.qq.com/login"], result.candidate_urls)


if __name__ == "__main__":
    unittest.main()
