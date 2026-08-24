import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


def sample_browser_task_envelope():
    from app.agent_runtime.contracts.handoff import HandoffPayloadBuilder
    from app.agent_runtime.external_tasks.schemas import (
        ExternalTaskCandidateProfileRef,
        ExternalTaskJobContext,
        FindApplyEntryTaskEnvelope,
    )

    return HandoffPayloadBuilder().build_browser_application_task(
        find_apply_entry_envelope=FindApplyEntryTaskEnvelope(
            task_id="browser-result-task-1",
            trace_id="trace-browser-result-1",
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
    )


class TaskContractResultsTest(unittest.TestCase):
    def test_browser_execution_result_converts_to_result_envelope(self) -> None:
        from app.agent_runtime.contracts.base import ArtifactRef
        from app.agent_runtime.contracts.results import browser_execution_result_to_result_envelope
        from app.agent_runtime.contracts.tasks.browser_application import BrowserExecutionResult, BrowserExecutionStatus
        from app.agent_runtime.tool_registry import APPLICATION_FIND_APPLY_ENTRY_TOOL

        result = BrowserExecutionResult(
            task_id="browser-result-task-1",
            status=BrowserExecutionStatus.WAITING_USER,
            summary="已打开腾讯申请页并停在提交前。",
            observations=["申请页可访问", "需要用户确认最终提交"],
            artifacts=[ArtifactRef(type="screenshot", title="申请页截图", url="F:/tmp/apply.png", mime_type="image/png")],
            requires_user_action=True,
            next_action="wait_user_review",
            current_url="https://careers.tencent.com/apply/1",
            apply_url="https://careers.tencent.com/apply/1",
        )

        envelope = browser_execution_result_to_result_envelope(
            task_envelope=sample_browser_task_envelope(),
            result=result,
            executor_name="codex_or_multica",
        ).to_dict()

        self.assertEqual("waiting_user", envelope["status"])
        self.assertEqual(APPLICATION_FIND_APPLY_ENTRY_TOOL, envelope["capability"])
        self.assertEqual("codex_or_multica", envelope["executor"])
        self.assertEqual("已打开腾讯申请页并停在提交前。", envelope["summary"])
        self.assertTrue(envelope["requires_user_action"])
        self.assertEqual("medium", envelope["risk_level"])
        self.assertEqual(["申请页可访问", "需要用户确认最终提交"], envelope["observations"])
        self.assertEqual(
            [{"type": "screenshot", "title": "申请页截图", "url": "F:/tmp/apply.png", "mime_type": "image/png", "metadata": {}}],
            envelope["artifacts"],
        )


if __name__ == "__main__":
    unittest.main()
