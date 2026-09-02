import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class DurableStateModelsTest(unittest.TestCase):
    def test_metadata_contains_durable_state_tables_and_relationships(self) -> None:
        import app.agent_runtime.durable_state.models  # noqa: F401
        from app.db.base import Base

        self.assertIn("agent_task_states", Base.metadata.tables)
        self.assertIn("agent_step_states", Base.metadata.tables)
        self.assertIn("agent_memory_snapshots", Base.metadata.tables)
        self.assertIn("agent_artifact_index", Base.metadata.tables)
        task_table = Base.metadata.tables["agent_task_states"]
        step_table = Base.metadata.tables["agent_step_states"]
        snapshot_table = Base.metadata.tables["agent_memory_snapshots"]
        artifact_table = Base.metadata.tables["agent_artifact_index"]

        self.assertIn("current_step_id", task_table.c)
        self.assertIn("output_payload", step_table.c)
        self.assertIn("passed_to_executor", snapshot_table.c)
        self.assertIn("artifact_metadata", artifact_table.c)
        foreign_keys = {foreign_key.target_fullname for foreign_key in step_table.c.task_id.foreign_keys}
        self.assertEqual({"agent_task_states.id"}, foreign_keys)


if __name__ == "__main__":
    unittest.main()
