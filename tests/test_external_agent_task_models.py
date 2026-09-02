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


class ExternalAgentTaskModelsTest(unittest.TestCase):
    def test_external_agent_task_tables_are_registered(self):
        from app.agent_runtime.external_tasks import models as external_task_models  # noqa: F401
        from app.db.base import Base

        expected_tables = {
            "external_agent_tasks",
            "external_agent_task_events",
            "external_agent_artifacts",
        }
        self.assertTrue(expected_tables.issubset(set(Base.metadata.tables)))

        tasks = Base.metadata.tables["external_agent_tasks"]
        events = Base.metadata.tables["external_agent_task_events"]
        artifacts = Base.metadata.tables["external_agent_artifacts"]

        self.assertEqual(
            {
                "id",
                "task_type",
                "status",
                "trace_id",
                "context_pack_hash",
                "input_payload",
                "output_payload",
                "blocked_reason",
                "created_at",
                "updated_at",
                "completed_at",
            },
            set(tasks.columns.keys()),
        )
        self.assertEqual(
            {"id", "task_id", "event_type", "payload", "created_at"},
            set(events.columns.keys()),
        )
        self.assertEqual(
            {
                "id",
                "task_id",
                "artifact_type",
                "path_or_uri",
                "mime_type",
                "artifact_metadata",
                "created_at",
            },
            set(artifacts.columns.keys()),
        )
        self.assertEqual(
            "external_agent_tasks.id",
            str(next(iter(events.c.task_id.foreign_keys)).column),
        )
        self.assertEqual(
            "external_agent_tasks.id",
            str(next(iter(artifacts.c.task_id.foreign_keys)).column),
        )

    def test_tenth_migration_creates_external_agent_task_tables(self):
        migration = (
            PROJECT_ROOT
            / "infra"
            / "migrations"
            / "versions"
            / "20260818_0010_external_agent_tasks.py"
        )
        self.assertTrue(migration.is_file())

        from app.core.config import get_settings

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as tmp_dir:
            db_path = Path(tmp_dir) / "external_agent_tasks_check.sqlite"
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
                    "external_agent_tasks",
                    "external_agent_task_events",
                    "external_agent_artifacts",
                }.issubset(set(inspector.get_table_names()))
            )
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
