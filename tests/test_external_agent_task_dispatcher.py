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


def sample_envelope():
    from app.agent_runtime.external_tasks.schemas import (
        ExternalTaskCandidateProfileRef,
        ExternalTaskJobContext,
        FindApplyEntryTaskEnvelope,
    )

    return FindApplyEntryTaskEnvelope(
        task_id="task-dispatch-1",
        trace_id="trace-dispatch-1",
        job=ExternalTaskJobContext(
            job_id="lead-dispatch-1",
            company_name="Tencent",
            title="Backend Engineer Intern",
            source_url="https://careers.tencent.com/job/1",
        ),
        candidate_profile_ref=ExternalTaskCandidateProfileRef(
            profile_id="default",
            resume_version_id="resume-v3",
        ),
    )


class FakeExecutor:
    executor_name = "fake-executor"

    def __init__(self, result=None, exc: Exception | None = None) -> None:
        self.result = result
        self.exc = exc
        self.seen_envelope = None

    def execute_find_apply_entry(self, envelope):
        self.seen_envelope = envelope
        if self.exc is not None:
            raise self.exc
        return self.result


class ExternalTaskDispatcherTest(unittest.TestCase):
    def test_dispatch_find_apply_entry_marks_running_and_records_executor_result(self):
        from app.agent_runtime.external_tasks.dispatcher import ExternalTaskDispatcher
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
        executor_result = ApplyEntryDiscoveryResult(
            task_id="task-dispatch-1",
            status=ApplyEntryDiscoveryStatus.FOUND_OPENED,
            confidence=0.9,
            apply_url="https://careers.tencent.com/apply/1",
            final_browser_url="https://careers.tencent.com/apply/1",
            evidence_artifacts=[
                ExternalAgentArtifactRef(
                    artifact_type="web_search_result",
                    path_or_uri="https://careers.tencent.com/apply/1",
                )
            ],
            next_action="wait_user_review",
        )
        executor = FakeExecutor(result=executor_result)

        dispatch = ExternalTaskDispatcher(repository=repository, executor=executor).dispatch("task-dispatch-1")

        self.assertTrue(dispatch.ok)
        self.assertEqual("fake-executor", dispatch.executor_name)
        self.assertEqual("lead-dispatch-1", executor.seen_envelope.job.job_id)
        task = repository.tasks["task-dispatch-1"]
        self.assertEqual(ExternalAgentTaskStatus.SUCCEEDED, task.status)
        self.assertEqual("https://careers.tencent.com/apply/1", task.output_payload["apply_url"])
        self.assertEqual("fake-executor", task.output_payload["result_envelope"]["executor"])
        self.assertEqual("applications.find_apply_entry", task.output_payload["result_envelope"]["capability"])
        self.assertIn("Tencent - Backend Engineer Intern", task.output_payload["result_envelope"]["summary"])
        self.assertEqual(task.output_payload["result_envelope"], dispatch.result_envelope)
        self.assertEqual(task.output_payload["result_envelope"], dispatch.to_dict()["result_envelope"])
        self.assertEqual(
            ["task_queued", "task_running", "task_succeeded"],
            [event["event_type"] for event in repository.events],
        )

    def test_dispatch_find_apply_entry_records_failed_result_when_executor_raises(self):
        from app.agent_runtime.external_tasks.dispatcher import ExternalTaskDispatcher
        from app.agent_runtime.external_tasks.schemas import ExternalAgentTaskStatus
        from app.agent_runtime.external_tasks.service import ExternalAgentTaskService

        repository = InMemoryExternalTaskRepository()
        ExternalAgentTaskService(repository).create_find_apply_entry_task(sample_envelope())

        dispatch = ExternalTaskDispatcher(
            repository=repository,
            executor=FakeExecutor(exc=RuntimeError("network unavailable")),
        ).dispatch("task-dispatch-1")

        self.assertFalse(dispatch.ok)
        self.assertIn("network unavailable", dispatch.error)
        task = repository.tasks["task-dispatch-1"]
        self.assertEqual(ExternalAgentTaskStatus.FAILED, task.status)
        self.assertEqual("failed", task.output_payload["status"])
        self.assertIn("network unavailable", task.output_payload["notes"])
        self.assertEqual("failed", task.output_payload["result_envelope"]["status"])
        self.assertEqual("fake-executor", task.output_payload["result_envelope"]["executor"])
        self.assertEqual(task.output_payload["result_envelope"], dispatch.result_envelope)
        self.assertEqual(
            ["task_queued", "task_running", "task_failed"],
            [event["event_type"] for event in repository.events],
        )


if __name__ == "__main__":
    unittest.main()
