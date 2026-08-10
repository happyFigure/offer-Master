import sys
import unittest
from pathlib import Path

from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class DatabaseEnvironmentTest(unittest.TestCase):
    def test_settings_defaults_to_local_mysql_database(self):
        from app.core.config import Settings

        settings = Settings(_env_file=None)

        self.assertEqual("mysql+pymysql", settings.database_url.drivername)
        self.assertEqual("127.0.0.1", settings.database_url.host)
        self.assertEqual(3306, settings.database_url.port)
        self.assertEqual("offermaster", settings.database_url.database)
        self.assertEqual(PROJECT_ROOT / "data" / "vector_store", settings.vector_store_path)

    def test_session_factory_runs_sql_against_bound_engine(self):
        from app.db.session import create_engine_from_url, create_session_factory

        engine = create_engine_from_url("sqlite+pysqlite:///:memory:")
        session_factory = create_session_factory(engine)

        with session_factory() as session:
            result = session.execute(text("select 1")).scalar_one()

        self.assertEqual(1, result)

    def test_unit_of_work_commits_and_closes_session(self):
        from app.db.session import create_engine_from_url, create_session_factory
        from app.db.unit_of_work import UnitOfWork

        engine = create_engine_from_url("sqlite+pysqlite:///:memory:")
        session_factory = create_session_factory(engine)

        with UnitOfWork(session_factory) as uow:
            self.assertEqual(1, uow.session.execute(text("select 1")).scalar_one())
            uow.commit()

        self.assertIsNone(uow.session)

    def test_alembic_environment_is_configured(self):
        self.assertTrue((PROJECT_ROOT / "alembic.ini").is_file())
        self.assertTrue((PROJECT_ROOT / "infra" / "migrations" / "env.py").is_file())
        self.assertTrue((PROJECT_ROOT / "infra" / "migrations" / "versions").is_dir())


if __name__ == "__main__":
    unittest.main()
