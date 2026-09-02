import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class DurableStateMigrationTest(unittest.TestCase):
    def test_eleventh_migration_creates_agent_durable_state_tables(self) -> None:
        migration = (
            PROJECT_ROOT
            / "infra"
            / "migrations"
            / "versions"
            / "20260821_0011_agent_durable_state.py"
        )
        self.assertTrue(migration.is_file())

        from app.core.config import get_settings

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as tmp_dir:
            db_path = Path(tmp_dir) / "agent_durable_state_check.sqlite"
            config = Config(str(PROJECT_ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(PROJECT_ROOT / "infra" / "migrations"))

            with patch.dict(
                "os.environ",
                {"JOBPILOT_DATABASE_URL": f"sqlite+pysqlite:///{db_path.as_posix()}"},
                clear=False,
            ):
                get_settings.cache_clear()
                command.upgrade(config, "head")

            engine = create_engine(f"sqlite+pysqlite:///{db_path.as_posix()}", future=True)
            inspector = inspect(engine)
            self.assertTrue(
                {
                    "agent_task_states",
                    "agent_step_states",
                }.issubset(set(inspector.get_table_names()))
            )
            task_columns = {column["name"] for column in inspector.get_columns("agent_task_states")}
            step_columns = {column["name"] for column in inspector.get_columns("agent_step_states")}
            self.assertIn("current_step_id", task_columns)
            self.assertIn("external_task_id", step_columns)
            self.assertIn("approval_request_id", step_columns)
            engine.dispose()

    def test_twelfth_migration_creates_memory_snapshot_and_artifact_tables(self) -> None:
        migration = (
            PROJECT_ROOT
            / "infra"
            / "migrations"
            / "versions"
            / "20260821_0012_agent_durable_state_snapshots.py"
        )
        self.assertTrue(migration.is_file())

        from app.core.config import get_settings

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as tmp_dir:
            db_path = Path(tmp_dir) / "agent_durable_state_snapshot_check.sqlite"
            config = Config(str(PROJECT_ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(PROJECT_ROOT / "infra" / "migrations"))

            with patch.dict(
                "os.environ",
                {"JOBPILOT_DATABASE_URL": f"sqlite+pysqlite:///{db_path.as_posix()}"},
                clear=False,
            ):
                get_settings.cache_clear()
                command.upgrade(config, "head")

            engine = create_engine(f"sqlite+pysqlite:///{db_path.as_posix()}", future=True)
            inspector = inspect(engine)
            self.assertTrue(
                {
                    "agent_memory_snapshots",
                    "agent_artifact_index",
                }.issubset(set(inspector.get_table_names()))
            )
            snapshot_columns = {column["name"] for column in inspector.get_columns("agent_memory_snapshots")}
            artifact_columns = {column["name"] for column in inspector.get_columns("agent_artifact_index")}
            self.assertIn("passed_to_executor", snapshot_columns)
            self.assertIn("visibility_scope", snapshot_columns)
            self.assertIn("artifact_metadata", artifact_columns)
            self.assertIn("uri", artifact_columns)
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
