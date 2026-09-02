import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class UrlImportRuntimeModelsTest(unittest.TestCase):
    def test_url_import_and_domain_health_tables_are_registered(self):
        from app.db.base import Base
        from app.domains.applications import models as application_models  # noqa: F401
        from app.domains.automation import models as automation_models  # noqa: F401
        from app.domains.jobs import models as job_models  # noqa: F401
        from app.domains.jobs.models import DomainHealthState, UrlImportRunStatus

        self.assertTrue(
            {"url_import_runs", "domain_health_states"}.issubset(set(Base.metadata.tables))
        )

        url_import_runs = Base.metadata.tables["url_import_runs"]
        domain_health_states = Base.metadata.tables["domain_health_states"]

        self.assertEqual(
            {
                "id",
                "workflow_run_id",
                "source_id",
                "input_url",
                "normalized_url",
                "normalized_url_hash",
                "source_type",
                "domain",
                "fetch_layer",
                "status",
                "current_stage",
                "attempt_count",
                "tool_call_count",
                "llm_call_count",
                "error_code",
                "error_message",
                "next_action",
                "raw_job_lead_id",
                "extracted_count",
                "duplicate_of_run_id",
                "run_metadata",
                "started_at",
                "updated_at",
                "finished_at",
            },
            set(url_import_runs.columns.keys()),
        )
        self.assertEqual(
            {
                "id",
                "domain",
                "tool_name",
                "state",
                "failure_count",
                "success_count",
                "last_error_code",
                "last_error_message",
                "opened_at",
                "cooldown_until",
                "half_open_probe_count",
                "created_at",
                "updated_at",
            },
            set(domain_health_states.columns.keys()),
        )
        self.assertEqual(
            "workflow_runs.id",
            str(next(iter(url_import_runs.c.workflow_run_id.foreign_keys)).column),
        )
        self.assertEqual(
            "job_sources.id",
            str(next(iter(url_import_runs.c.source_id.foreign_keys)).column),
        )
        self.assertIn("failed_recoverable", {status.value for status in UrlImportRunStatus})
        self.assertEqual("half_open", DomainHealthState.HALF_OPEN.value)

    def test_can_persist_url_import_run_and_domain_health_state(self):
        from app.db.base import Base
        from app.domains.applications import models as application_models  # noqa: F401
        from app.domains.automation.models import WorkflowRun, WorkflowRunStatus
        from app.domains.jobs.models import (
            DomainHealth,
            DomainHealthState,
            JobSource,
            JobSourceFetchMode,
            JobSourceTrustLevel,
            JobSourceType,
            UrlImportRun,
            UrlImportRunStatus,
        )

        engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)

        with Session() as session:
            source = JobSource(
                name="公众号秋招汇总",
                source_type=JobSourceType.WECHAT_ARTICLE,
                entry_url="https://mp.weixin.qq.com/s/example",
                trust_level=JobSourceTrustLevel.MEDIUM_HIGH,
                fetch_mode=JobSourceFetchMode.PUBLIC_HTML,
            )
            workflow = WorkflowRun(
                id="workflow-run-url-import-1",
                workflow_type="url_import",
                status=WorkflowRunStatus.RUNNING,
                current_step="wechat_article_fetch",
                user_goal="Import recruiting leads from URL",
            )
            run = UrlImportRun(
                workflow_run_id="workflow-run-url-import-1",
                source=source,
                input_url="https://mp.weixin.qq.com/s/example?utm_source=xhs",
                normalized_url="https://mp.weixin.qq.com/s/example",
                normalized_url_hash="abc123",
                source_type=JobSourceType.WECHAT_ARTICLE,
                domain="mp.weixin.qq.com",
                fetch_layer="wechat_article",
                status=UrlImportRunStatus.RUNNING,
                current_stage="wechat_article_fetch",
                attempt_count=1,
                tool_call_count=2,
                llm_call_count=0,
                next_action="wait_for_fetch_result",
                run_metadata={"source_hint": "wechat_article"},
            )
            health = DomainHealth(
                domain="mp.weixin.qq.com",
                tool_name="WeChatArticleFetcher",
                state=DomainHealthState.OPEN,
                failure_count=5,
                success_count=18,
                last_error_code="HTTP_429",
                last_error_message="too many requests",
                opened_at=datetime.now(UTC).replace(tzinfo=None),
                cooldown_until=datetime.now(UTC).replace(tzinfo=None) + timedelta(minutes=30),
                half_open_probe_count=0,
            )
            session.add_all([workflow, source, run, health])
            session.commit()

            persisted_run = session.query(UrlImportRun).one()
            persisted_health = session.query(DomainHealth).one()

        self.assertEqual(UrlImportRunStatus.RUNNING, persisted_run.status)
        self.assertEqual("wechat_article_fetch", persisted_run.current_stage)
        self.assertEqual("公众号秋招汇总", persisted_run.source.name)
        self.assertEqual(DomainHealthState.OPEN, persisted_health.state)
        self.assertEqual("HTTP_429", persisted_health.last_error_code)

    def test_fourth_migration_creates_url_import_runtime_tables(self):
        migration = (
            PROJECT_ROOT
            / "infra"
            / "migrations"
            / "versions"
            / "20260812_0004_url_import_runtime_tables.py"
        )
        self.assertTrue(migration.is_file())

        from app.core.config import get_settings

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as tmp_dir:
            db_path = Path(tmp_dir) / "url_import_runtime_check.sqlite"
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
                {"url_import_runs", "domain_health_states"}.issubset(
                    set(inspector.get_table_names())
                )
            )
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
