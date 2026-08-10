import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class RuntimeModelsTest(unittest.TestCase):
    def test_runtime_tables_are_registered_with_recovery_relationships(self):
        from app.db.base import Base
        from app.domains.automation.models import WorkflowRunStatus

        expected_tables = {
            "workflow_runs",
            "workflow_checkpoints",
            "tool_call_logs",
            "approval_requests",
            "automation_runs",
        }
        self.assertTrue(expected_tables.issubset(set(Base.metadata.tables)))

        workflow_runs = Base.metadata.tables["workflow_runs"]
        checkpoints = Base.metadata.tables["workflow_checkpoints"]
        tool_logs = Base.metadata.tables["tool_call_logs"]
        approvals = Base.metadata.tables["approval_requests"]
        automation_runs = Base.metadata.tables["automation_runs"]

        self.assertEqual(
            {
                "id",
                "workflow_type",
                "status",
                "current_step",
                "user_goal",
                "related_job_id",
                "related_application_id",
                "approval_request_id",
                "error",
                "started_at",
                "updated_at",
                "completed_at",
            },
            set(workflow_runs.columns.keys()),
        )
        self.assertEqual(
            {
                "id",
                "workflow_run_id",
                "checkpoint_key",
                "state",
                "created_at",
            },
            set(checkpoints.columns.keys()),
        )
        self.assertEqual(
            {
                "id",
                "workflow_run_id",
                "tool_name",
                "tool_group",
                "status",
                "input_payload",
                "output_payload",
                "error",
                "duration_ms",
                "created_at",
            },
            set(tool_logs.columns.keys()),
        )
        self.assertEqual(
            {
                "id",
                "workflow_run_id",
                "application_id",
                "action_type",
                "status",
                "prompt",
                "payload",
                "decision",
                "decided_at",
                "created_at",
                "expires_at",
            },
            set(approvals.columns.keys()),
        )
        self.assertEqual(
            {
                "id",
                "workflow_run_id",
                "application_id",
                "browser_session_id",
                "status",
                "target_url",
                "last_screenshot_path",
                "error",
                "started_at",
                "updated_at",
                "completed_at",
            },
            set(automation_runs.columns.keys()),
        )
        self.assertEqual(
            "workflow_runs.id",
            str(next(iter(checkpoints.c.workflow_run_id.foreign_keys)).column),
        )
        self.assertIn("waiting_user", {status.value for status in WorkflowRunStatus})

    def test_can_persist_checkpoint_tool_log_approval_and_automation_run(self):
        from app.db.base import Base
        from app.domains.applications.models import Application, ApplicationStatus
        from app.domains.automation.models import (
            ApprovalRequest,
            ApprovalRequestStatus,
            AutomationRun,
            AutomationRunStatus,
            ToolCallLog,
            ToolCallStatus,
            WorkflowCheckpoint,
            WorkflowRun,
            WorkflowRunStatus,
        )
        from app.domains.jobs.models import Company, Job, JobStatus

        engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)

        with Session() as session:
            company = Company(name="Runtime Inc", normalized_name="runtime inc")
            job = Job(
                company=company,
                title="Backend Agent Engineer",
                source="mock",
                source_job_id="runtime-001",
                skills=[],
                status=JobStatus.OPEN,
            )
            application = Application(job=job, status=ApplicationStatus.PREPARING)
            run = WorkflowRun(
                workflow_type="application_assist",
                status=WorkflowRunStatus.WAITING_USER,
                current_step="confirm_submit",
                user_goal="Assist application submission",
                related_job=job,
                related_application=application,
            )
            checkpoint = WorkflowCheckpoint(
                workflow_run=run,
                checkpoint_key="confirm_submit",
                state={"current_step": "confirm_submit"},
            )
            tool_log = ToolCallLog(
                workflow_run=run,
                tool_name="fill_form",
                tool_group="mcp",
                status=ToolCallStatus.SUCCEEDED,
                input_payload={"field": "name"},
                output_payload={"filled": True},
                duration_ms=123,
            )
            approval = ApprovalRequest(
                workflow_run=run,
                application=application,
                action_type="submit_application",
                status=ApprovalRequestStatus.PENDING,
                prompt="Confirm real submission",
                payload={"url": "https://example.com/apply"},
            )
            automation = AutomationRun(
                workflow_run=run,
                application=application,
                browser_session_id="edge-session-1",
                status=AutomationRunStatus.WAITING_USER,
                target_url="https://example.com/apply",
                last_screenshot_path="F:/pythonProject/OfferMaster/data/exports/apply.png",
            )
            run.approval_request = approval

            session.add_all([checkpoint, tool_log, automation])
            session.commit()

            persisted = session.query(WorkflowRun).one()

        self.assertEqual(WorkflowRunStatus.WAITING_USER, persisted.status)
        self.assertEqual("confirm_submit", persisted.checkpoints[0].checkpoint_key)
        self.assertEqual({"filled": True}, persisted.tool_call_logs[0].output_payload)
        self.assertEqual("submit_application", persisted.approval_request.action_type)
        self.assertEqual(AutomationRunStatus.WAITING_USER, persisted.automation_runs[0].status)
        self.assertEqual("Runtime Inc", persisted.related_job.company.name)

    def test_second_migration_creates_runtime_tables_after_core_tables(self):
        migration = PROJECT_ROOT / "infra" / "migrations" / "versions" / "20260810_0002_runtime_tables.py"
        self.assertTrue(migration.is_file())

        from app.core.config import get_settings

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "data") as tmp_dir:
            db_path = Path(tmp_dir) / "runtime_migration_check.sqlite"
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
                    "workflow_runs",
                    "workflow_checkpoints",
                    "tool_call_logs",
                    "approval_requests",
                    "automation_runs",
                }.issubset(set(inspector.get_table_names()))
            )
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
