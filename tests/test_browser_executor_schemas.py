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
        task_id="browser-task-1",
        trace_id="trace-browser-1",
        job=ExternalTaskJobContext(
            job_id="job-1",
            company_name="Tencent",
            title="Backend Engineer Intern",
            source_url="https://careers.tencent.com/job/1",
            apply_url_candidate="https://careers.tencent.com/apply/1",
            jd_summary="Campus backend role requiring Java.",
        ),
        candidate_profile_ref=ExternalTaskCandidateProfileRef(
            profile_id="default",
            resume_version_id="resume-v3",
        ),
    )


class BrowserExecutorSchemasTest(unittest.TestCase):
    def test_browser_task_envelope_from_apply_entry_task_preserves_safe_stop_rules(self) -> None:
        from app.agent_runtime.external_tasks.schemas import BrowserTaskEnvelope, BrowserTaskType

        envelope = BrowserTaskEnvelope.from_find_apply_entry_task(sample_find_apply_entry_envelope())

        self.assertEqual("browser-task-1", envelope.task_id)
        self.assertEqual(BrowserTaskType.PREPARE_APPLICATION, envelope.task_type)
        self.assertEqual("https://careers.tencent.com/apply/1", envelope.start_url)
        self.assertTrue(envelope.stop_before_submit)
        self.assertIn("open_apply_page", envelope.allowed_actions)
        self.assertIn("fill_common_fields", envelope.allowed_actions)
        self.assertIn("generate_question_answers", envelope.allowed_actions)
        self.assertIn("submit_application", envelope.forbidden_actions)
        self.assertIn("final_submit", envelope.human_approval_required)
        self.assertIsNone(envelope.selected_resume_file_ref)


if __name__ == "__main__":
    unittest.main()
