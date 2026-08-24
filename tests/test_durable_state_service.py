import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class DurableStateServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        import app.agent_runtime.durable_state.models  # noqa: F401
        from app.db.base import Base

        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_service_tracks_task_current_step_and_status_transitions(self) -> None:
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.schemas import AgentStepStatus, AgentTaskStatus
        from app.agent_runtime.durable_state.service import DurableStateService

        with self.Session() as session:
            service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            task = service.create_task(
                task_id="task-1",
                root_workflow_run_id="workflow-1",
                conversation_session_id="session-1",
                task_type="application_orchestration",
                capability="applications.find_apply_entry",
                user_goal="打开腾讯申请页，停在提交前",
            )
            step = service.add_step(
                task_id=task.id,
                step_id="step-1",
                sequence_index=1,
                step_type="browser_executor",
                executor_type="browser_executor",
                executor_name="codex_or_multica",
                capability="applications.find_apply_entry",
                input_payload={"job_id": "job-1"},
            )

            self.assertEqual("step-1", service.get_task("task-1").current_step_id)
            self.assertEqual(AgentTaskStatus.CREATED, service.get_task("task-1").status)

            service.mark_step_running("step-1")
            self.assertEqual(AgentStepStatus.RUNNING, service.get_step("step-1").status)
            self.assertEqual(AgentTaskStatus.RUNNING, service.get_task("task-1").status)

            service.mark_step_waiting_user("step-1", approval_request_id="approval-1", output_payload={"next_action": "wait_user_login"})
            self.assertEqual(AgentStepStatus.WAITING_USER, service.get_step("step-1").status)
            self.assertEqual(AgentTaskStatus.WAITING_USER, service.get_task("task-1").status)
            self.assertEqual("approval-1", service.get_step("step-1").approval_request_id)

            service.mark_step_succeeded(
                "step-1",
                tool_call_log_id="tool-log-1",
                external_task_id="external-task-1",
                output_payload={"apply_url": "https://careers.tencent.com/apply/1"},
            )

            self.assertEqual(AgentStepStatus.SUCCEEDED, service.get_step("step-1").status)
            self.assertEqual(AgentTaskStatus.RUNNING, service.get_task("task-1").status)
            self.assertEqual("tool-log-1", service.get_step("step-1").tool_call_log_id)
            self.assertEqual("external-task-1", service.get_step("step-1").external_task_id)
            self.assertEqual({"apply_url": "https://careers.tencent.com/apply/1"}, service.get_step("step-1").output_payload)

    def test_service_marks_task_failed_when_step_fails(self) -> None:
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.schemas import AgentTaskStatus
        from app.agent_runtime.durable_state.service import DurableStateService

        with self.Session() as session:
            service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            task = service.create_task(
                task_id="task-2",
                root_workflow_run_id="workflow-2",
                conversation_session_id="session-2",
                task_type="application_orchestration",
                capability="applications.find_apply_entry",
                user_goal="打开腾讯申请页，停在提交前",
            )
            service.add_step(
                task_id=task.id,
                step_id="step-2",
                sequence_index=1,
                step_type="browser_executor",
                executor_type="browser_executor",
                executor_name="codex_or_multica",
                capability="applications.find_apply_entry",
            )

            service.mark_step_failed("step-2", output_payload={"error": "executor unavailable"})

            self.assertEqual(AgentTaskStatus.FAILED, service.get_task("task-2").status)
            self.assertEqual({"error": "executor unavailable"}, service.get_step("step-2").output_payload)


if __name__ == "__main__":
    unittest.main()
