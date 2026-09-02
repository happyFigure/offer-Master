import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class InMemoryExternalTaskRepository:
    def __init__(self) -> None:
        self.tasks = {}
        self.events = []
        self.artifacts = {}

    def create(self, task):
        self.tasks[task.task_id] = task
        return task

    def get(self, task_id):
        return self.tasks.get(task_id)

    def save(self, task):
        self.tasks[task.task_id] = task
        return task

    def add_event(self, task_id, event_type, payload):
        self.events.append({"task_id": task_id, "event_type": event_type, "payload": payload})

    def replace_artifacts(self, task_id, artifacts):
        self.artifacts[task_id] = artifacts


def sample_envelope(task_id="task-001"):
    from app.agent_runtime.external_tasks.schemas import (
        ExternalTaskCandidateProfileRef,
        ExternalTaskJobContext,
        FindApplyEntryTaskEnvelope,
    )

    return FindApplyEntryTaskEnvelope(
        task_id=task_id,
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


class ExternalAgentTaskServiceTest(unittest.TestCase):
    def test_create_find_apply_entry_task_stores_queued_task_with_context_hash(self):
        from app.agent_runtime.external_tasks.schemas import ExternalAgentTaskStatus, ExternalTaskType
        from app.agent_runtime.external_tasks.service import ExternalAgentTaskService

        repository = InMemoryExternalTaskRepository()
        task = ExternalAgentTaskService(repository).create_find_apply_entry_task(sample_envelope())

        self.assertEqual("task-001", task.task_id)
        self.assertEqual(ExternalTaskType.FIND_APPLY_ENTRY, task.task_type)
        self.assertEqual(ExternalAgentTaskStatus.QUEUED, task.status)
        self.assertEqual(64, len(task.context_pack_hash))
        self.assertEqual("job-001", task.input_payload["job"]["job_id"])
        self.assertIsNone(task.output_payload)
        self.assertEqual("task_queued", repository.events[0]["event_type"])

    def test_mark_running_records_state_transition_event(self):
        from app.agent_runtime.external_tasks.schemas import ExternalAgentTaskStatus
        from app.agent_runtime.external_tasks.service import ExternalAgentTaskService

        repository = InMemoryExternalTaskRepository()
        service = ExternalAgentTaskService(repository)
        service.create_find_apply_entry_task(sample_envelope())

        task = service.mark_running("task-001")

        self.assertEqual(ExternalAgentTaskStatus.RUNNING, task.status)
        self.assertEqual("task_running", repository.events[-1]["event_type"])

    def test_record_found_opened_result_succeeds_and_persists_output(self):
        from app.agent_runtime.external_tasks.schemas import (
            ApplyEntryDiscoveryResult,
            ApplyEntryDiscoveryStatus,
            ExternalAgentArtifactRef,
            ExternalAgentTaskStatus,
        )
        from app.agent_runtime.external_tasks.service import ExternalAgentTaskService

        repository = InMemoryExternalTaskRepository()
        service = ExternalAgentTaskService(repository)
        service.create_find_apply_entry_task(sample_envelope())

        result = ApplyEntryDiscoveryResult(
            task_id="task-001",
            status=ApplyEntryDiscoveryStatus.FOUND_OPENED,
            apply_url="https://join.qq.com/campus/apply/001",
            final_browser_url="https://join.qq.com/campus/apply/001/form",
            confidence=0.92,
            evidence_artifacts=[ExternalAgentArtifactRef(artifact_type="screenshot", path_or_uri="F:/tmp/apply.png")],
            next_action="wait_user_review",
        )
        task = service.record_result("task-001", result, executor_name="codex_or_multica")

        self.assertEqual(ExternalAgentTaskStatus.SUCCEEDED, task.status)
        self.assertEqual("found_opened", task.output_payload["status"])
        self.assertEqual("https://join.qq.com/campus/apply/001", task.output_payload["apply_url"])
        self.assertEqual("codex_or_multica", task.output_payload["result_envelope"]["executor"])
        self.assertEqual("applications.find_apply_entry", task.output_payload["result_envelope"]["capability"])
        self.assertTrue(task.output_payload["result_envelope"]["requires_user_action"])
        self.assertIsNone(task.blocked_reason)
        self.assertEqual("task_succeeded", repository.events[-1]["event_type"])
        self.assertEqual("screenshot", repository.artifacts["task-001"][0]["artifact_type"])

    def test_record_blocked_result_moves_task_to_waiting_user(self):
        from app.agent_runtime.external_tasks.schemas import (
            ApplyEntryBlockedReason,
            ApplyEntryDiscoveryResult,
            ApplyEntryDiscoveryStatus,
            ExternalAgentTaskStatus,
        )
        from app.agent_runtime.external_tasks.service import ExternalAgentTaskService

        repository = InMemoryExternalTaskRepository()
        service = ExternalAgentTaskService(repository)
        service.create_find_apply_entry_task(sample_envelope())

        task = service.record_result(
            "task-001",
            ApplyEntryDiscoveryResult(
                task_id="task-001",
                status=ApplyEntryDiscoveryStatus.BLOCKED,
                blocked_reason=ApplyEntryBlockedReason.LOGIN_REQUIRED,
                confidence=0.5,
                candidate_urls=["https://join.qq.com/login"],
                next_action="ask_user_login",
            ),
        )

        self.assertEqual(ExternalAgentTaskStatus.WAITING_USER, task.status)
        self.assertEqual("login_required", task.blocked_reason)
        self.assertEqual("task_waiting_user", repository.events[-1]["event_type"])

    def test_record_result_rejects_mismatched_task_id(self):
        from app.agent_runtime.external_tasks.schemas import ApplyEntryDiscoveryResult, ApplyEntryDiscoveryStatus
        from app.agent_runtime.external_tasks.service import ExternalAgentTaskService

        repository = InMemoryExternalTaskRepository()
        service = ExternalAgentTaskService(repository)
        service.create_find_apply_entry_task(sample_envelope())

        with self.assertRaisesRegex(ValueError, "does not match"):
            service.record_result(
                "task-001",
                ApplyEntryDiscoveryResult(
                    task_id="other-task",
                    status=ApplyEntryDiscoveryStatus.FAILED,
                    confidence=0,
                    next_action="retry",
                ),
            )


if __name__ == "__main__":
    unittest.main()
