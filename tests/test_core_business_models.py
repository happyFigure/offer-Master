import sys
import tempfile
import unittest
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class CoreBusinessModelsTest(unittest.TestCase):
    def test_company_job_application_and_event_tables_are_registered(self):
        from app.db.base import Base
        from app.domains.applications.models import ApplicationStatus

        expected_tables = {
            "companies",
            "jobs",
            "applications",
            "application_events",
        }
        self.assertTrue(expected_tables.issubset(set(Base.metadata.tables)))

        companies = Base.metadata.tables["companies"]
        jobs = Base.metadata.tables["jobs"]
        applications = Base.metadata.tables["applications"]
        events = Base.metadata.tables["application_events"]

        self.assertEqual(
            {
                "id",
                "name",
                "normalized_name",
                "website_url",
                "industry",
                "city",
                "country",
                "raw_payload",
                "created_at",
                "updated_at",
            },
            set(companies.columns.keys()),
        )
        self.assertEqual(
            {
                "id",
                "company_id",
                "title",
                "city",
                "source",
                "source_job_id",
                "source_url",
                "job_type",
                "salary_text",
                "jd_text",
                "skills",
                "date_posted",
                "match_score",
                "status",
                "raw_payload",
                "created_at",
                "updated_at",
            },
            set(jobs.columns.keys()),
        )
        self.assertEqual(
            {
                "id",
                "job_id",
                "status",
                "priority",
                "channel",
                "applied_at",
                "next_follow_up_at",
                "notes",
                "created_at",
                "updated_at",
            },
            set(applications.columns.keys()),
        )
        self.assertEqual(
            {
                "id",
                "application_id",
                "event_type",
                "from_status",
                "to_status",
                "title",
                "body",
                "actor",
                "source",
                "event_metadata",
                "created_at",
            },
            set(events.columns.keys()),
        )

        self.assertEqual("companies.id", str(next(iter(jobs.c.company_id.foreign_keys)).column))
        self.assertEqual("jobs.id", str(next(iter(applications.c.job_id.foreign_keys)).column))
        self.assertEqual(
            "applications.id",
            str(next(iter(events.c.application_id.foreign_keys)).column),
        )
        self.assertIn(
            "applied",
            {status.value for status in ApplicationStatus},
        )

    def test_can_persist_application_timeline_for_imported_job(self):
        from app.db.base import Base
        from app.domains.applications.models import (
            Application,
            ApplicationEvent,
            ApplicationStatus,
        )
        from app.domains.jobs.models import Company, Job, JobStatus

        engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)

        with Session() as session:
            company = Company(name="Acme AI", normalized_name="acme ai")
            job = Job(
                company=company,
                title="AI Application Engineer",
                city="Beijing",
                source="mock",
                source_job_id="mock-001",
                source_url="https://example.com/jobs/mock-001",
                job_type="campus",
                salary_text="negotiable",
                jd_text="Build LLM and RAG applications with Python.",
                skills=["Python", "LLM", "RAG"],
                date_posted=date(2026, 8, 10),
                match_score=86.5,
                status=JobStatus.OPEN,
                raw_payload={"source": "fixture"},
            )
            application = Application(
                job=job,
                status=ApplicationStatus.PREPARING,
                priority="high",
                channel="manual",
                notes="Prepare tailored resume before applying.",
            )
            event = ApplicationEvent(
                application=application,
                event_type="status_changed",
                from_status=ApplicationStatus.EVALUATING,
                to_status=ApplicationStatus.PREPARING,
                title="Moved to preparing",
                body="User decided to prepare application materials.",
                actor="user",
                source="manual",
                event_metadata={"reason": "high match"},
                created_at=datetime(2026, 8, 10, 9, 30, tzinfo=UTC),
            )

            session.add(event)
            session.commit()

            persisted = session.query(Application).one()

        self.assertEqual("Acme AI", persisted.job.company.name)
        self.assertEqual(["Python", "LLM", "RAG"], persisted.job.skills)
        self.assertEqual(ApplicationStatus.PREPARING, persisted.status)
        self.assertEqual("Moved to preparing", persisted.events[0].title)
        self.assertEqual({"reason": "high match"}, persisted.events[0].event_metadata)

    def test_initial_migration_creates_core_business_tables(self):
        migration = PROJECT_ROOT / "infra" / "migrations" / "versions" / "20260810_0001_core_business_tables.py"
        self.assertTrue(migration.is_file())

        from app.core.config import get_settings

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as tmp_dir:
            db_path = Path(tmp_dir) / "migration_check.sqlite"
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
            self.assertEqual(
                {"alembic_version", "companies", "jobs", "applications", "application_events"},
                set(inspector.get_table_names()),
            )
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
