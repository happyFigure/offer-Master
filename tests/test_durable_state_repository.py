import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class DurableStateRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        import app.agent_runtime.durable_state.models  # noqa: F401
        from app.db.base import Base

        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_repository_creates_task_and_lists_ordered_steps(self) -> None:
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.schemas import AgentStepStateCreate, AgentTaskStateCreate

        with self.Session() as session:
            repository = SqlAlchemyDurableStateRepository(session)
            task = repository.create_task(
                AgentTaskStateCreate(
                    task_id="task-1",
                    root_workflow_run_id="workflow-1",
                    conversation_session_id="session-1",
                    task_type="application_orchestration",
                    capability="applications.find_apply_entry",
                    user_goal="打开腾讯申请页，停在提交前",
                )
            )
            step = repository.add_step(
                AgentStepStateCreate(
                    step_id="step-1",
                    task_id=task.id,
                    sequence_index=1,
                    step_type="browser_executor",
                    executor_type="browser_executor",
                    executor_name="codex_or_multica",
                    capability="applications.find_apply_entry",
                    input_payload={"job_id": "job-1"},
                )
            )
            session.commit()

        with self.Session() as session:
            repository = SqlAlchemyDurableStateRepository(session)
            persisted = repository.get_task("task-1")
            steps = repository.list_steps("task-1")

        self.assertIsNotNone(persisted)
        self.assertEqual("task-1", persisted.id)
        self.assertEqual("step-1", step.id)
        self.assertEqual(["step-1"], [item.id for item in steps])
        self.assertEqual({"job_id": "job-1"}, steps[0].input_payload)


if __name__ == "__main__":
    unittest.main()
