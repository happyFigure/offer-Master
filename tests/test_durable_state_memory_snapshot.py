import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class DurableStateMemorySnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        import app.agent_runtime.durable_state.models  # noqa: F401
        from app.db.base import Base

        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_service_records_memory_snapshot_for_one_step(self) -> None:
        from app.agent_runtime.durable_state.repository import SqlAlchemyDurableStateRepository
        from app.agent_runtime.durable_state.schemas import AgentMemoryVisibilityScope
        from app.agent_runtime.durable_state.service import DurableStateService

        with self.Session() as session:
            service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            service.create_task(
                task_id="task-memory-1",
                root_workflow_run_id="workflow-memory-1",
                conversation_session_id="session-memory-1",
                task_type="application_orchestration",
                capability="memory_search",
            )
            service.add_step(
                task_id="task-memory-1",
                step_id="step-memory-1",
                sequence_index=1,
                step_type="tool_call",
                executor_type="tool_registry",
                executor_name="agent_tool_registry",
                capability="memory_search",
            )

            snapshot = service.record_memory_snapshot(
                snapshot_id="snapshot-1",
                task_id="task-memory-1",
                step_id="step-memory-1",
                memory_id="memory-1",
                source_type="agent_memory",
                usage_reason="matched user Java backend preference",
                visibility_scope=AgentMemoryVisibilityScope.RUNTIME_ONLY,
                passed_to_executor=False,
                memory_payload={"title": "Java backend preference"},
            )
            session.commit()

        with self.Session() as session:
            service = DurableStateService(SqlAlchemyDurableStateRepository(session))
            snapshots = service.list_memory_snapshots("task-memory-1")

        self.assertEqual("snapshot-1", snapshot.id)
        self.assertEqual(["memory-1"], [item.memory_id for item in snapshots])
        self.assertEqual(AgentMemoryVisibilityScope.RUNTIME_ONLY, snapshots[0].visibility_scope)
        self.assertFalse(snapshots[0].passed_to_executor)
        self.assertEqual({"title": "Java backend preference"}, snapshots[0].memory_payload)


if __name__ == "__main__":
    unittest.main()
