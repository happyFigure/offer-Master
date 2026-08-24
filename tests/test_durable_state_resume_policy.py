import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class DurableStateResumePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        import app.agent_runtime.durable_state.models  # noqa: F401
        from app.db.base import Base

        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_resume_policy_waits_for_user_when_latest_step_needs_approval(self) -> None:
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.resume_policy import DurableResumeAction, DurableResumePolicy
        from app.agent_runtime.durable_state.service import DurableStateService

        with self.Session() as session:
            service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            service.create_task(
                task_id="task-resume-1",
                root_workflow_run_id="workflow-resume-1",
                conversation_session_id="session-resume-1",
                task_type="application_orchestration",
                capability="applications.find_apply_entry",
            )
            service.add_step(
                task_id="task-resume-1",
                step_id="step-resume-1",
                sequence_index=1,
                step_type="browser_executor",
                executor_type="browser_executor",
                executor_name="codex_or_multica",
                capability="applications.find_apply_entry",
            )
            service.mark_step_waiting_user(
                "step-resume-1",
                approval_request_id="approval-resume-1",
                output_payload={"next_action": "wait_user_login"},
            )
            decision = DurableResumePolicy().decide(
                service.get_task("task-resume-1"),
                service.list_steps("task-resume-1"),
            )

        self.assertEqual(DurableResumeAction.WAIT_USER_ACTION, decision.action)
        self.assertEqual("approval-resume-1", decision.approval_request_id)
        self.assertEqual("step-resume-1", decision.step_id)

    def test_resume_policy_retries_failed_retryable_step(self) -> None:
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.resume_policy import DurableResumeAction, DurableResumePolicy
        from app.agent_runtime.durable_state.service import DurableStateService

        with self.Session() as session:
            service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            service.create_task(
                task_id="task-resume-2",
                root_workflow_run_id="workflow-resume-2",
                conversation_session_id="session-resume-2",
                task_type="campus_search",
                capability="external.web_search",
            )
            step = service.add_step(
                task_id="task-resume-2",
                step_id="step-resume-2",
                sequence_index=1,
                step_type="external_agent",
                executor_type="external_agent",
                executor_name="claude_sdk_agent",
                capability="external.web_search",
            )
            step.retry_count = 1
            service.mark_step_failed("step-resume-2", output_payload={"error": "temporary executor timeout"})
            decision = DurableResumePolicy(max_retries=2).decide(
                service.get_task("task-resume-2"),
                service.list_steps("task-resume-2"),
            )

        self.assertEqual(DurableResumeAction.RETRY_FAILED_STEP, decision.action)
        self.assertEqual("step-resume-2", decision.step_id)

    def test_resume_policy_replans_when_failed_step_exhausts_retries(self) -> None:
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.resume_policy import DurableResumeAction, DurableResumePolicy
        from app.agent_runtime.durable_state.service import DurableStateService

        with self.Session() as session:
            service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            service.create_task(
                task_id="task-resume-3",
                root_workflow_run_id="workflow-resume-3",
                conversation_session_id="session-resume-3",
                task_type="campus_search",
                capability="external.web_search",
            )
            step = service.add_step(
                task_id="task-resume-3",
                step_id="step-resume-3",
                sequence_index=1,
                step_type="external_agent",
                executor_type="external_agent",
                executor_name="claude_sdk_agent",
                capability="external.web_search",
            )
            step.retry_count = 2
            service.mark_step_failed("step-resume-3", output_payload={"error": "repeated failure"})
            decision = DurableResumePolicy(max_retries=2).decide(
                service.get_task("task-resume-3"),
                service.list_steps("task-resume-3"),
            )

        self.assertEqual(DurableResumeAction.REPLAN_REMAINING_STEPS, decision.action)
        self.assertEqual("retry budget exhausted", decision.reason)

    def test_resume_task_creates_retry_step_entry_for_failed_step(self) -> None:
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.resume_policy import DurableResumeAction
        from app.agent_runtime.durable_state.schemas import AgentStepStatus, AgentTaskStatus
        from app.agent_runtime.durable_state.service import DurableStateService

        with self.Session() as session:
            service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            service.create_task(
                task_id="task-resume-entry",
                root_workflow_run_id="workflow-resume-entry",
                conversation_session_id="session-resume-entry",
                task_type="campus_search",
                capability="external.web_search",
                input_payload={"goal": "search campus recruiting"},
            )
            step = service.add_step(
                task_id="task-resume-entry",
                step_id="step-resume-entry-1",
                sequence_index=1,
                step_type="external_agent",
                executor_type="external_agent",
                executor_name="claude_sdk_agent",
                capability="external.web_search",
                input_payload={"query": "腾讯 校园招聘 官网"},
            )
            step.retry_count = 1
            service.mark_step_failed("step-resume-entry-1", output_payload={"error": "executor timeout"})

            result = service.resume_task("task-resume-entry", max_retries=2)
            task = service.get_task("task-resume-entry")
            steps = service.list_steps("task-resume-entry")
            session.commit()

        self.assertEqual(DurableResumeAction.RETRY_FAILED_STEP, result.action)
        self.assertEqual("step-resume-entry-1", result.source_step_id)
        self.assertIsNotNone(result.resume_step_id)
        self.assertEqual(result.resume_step_id, task.current_step_id)
        self.assertEqual(AgentTaskStatus.RUNNING, task.status)
        self.assertEqual(2, len(steps))
        retry_step = steps[-1]
        self.assertEqual(result.resume_step_id, retry_step.id)
        self.assertEqual("step-resume-entry-1", retry_step.parent_step_id)
        self.assertEqual(2, retry_step.sequence_index)
        self.assertEqual(AgentStepStatus.PENDING, retry_step.status)
        self.assertEqual(2, retry_step.retry_count)
        self.assertEqual({"query": "腾讯 校园招聘 官网"}, retry_step.input_payload)


if __name__ == "__main__":
    unittest.main()
