"""create runtime recovery tables

Revision ID: 20260810_0002
Revises: 20260810_0001
Create Date: 2026-08-10 00:10:00
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260810_0002"
down_revision: str | None = "20260810_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workflow_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("workflow_type", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("current_step", sa.String(length=128), nullable=True),
        sa.Column("user_goal", sa.Text(), nullable=True),
        sa.Column("related_job_id", sa.String(length=36), nullable=True),
        sa.Column("related_application_id", sa.String(length=36), nullable=True),
        sa.Column("approval_request_id", sa.String(length=36), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["related_application_id"], ["applications.id"]),
        sa.ForeignKeyConstraint(["related_job_id"], ["jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_workflow_runs_related_application_id", "workflow_runs", ["related_application_id"])
    op.create_index("ix_workflow_runs_related_job_id", "workflow_runs", ["related_job_id"])
    op.create_index("ix_workflow_runs_status", "workflow_runs", ["status"])
    op.create_index("ix_workflow_runs_workflow_type", "workflow_runs", ["workflow_type"])

    op.create_table(
        "workflow_checkpoints",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("workflow_run_id", sa.String(length=36), nullable=False),
        sa.Column("checkpoint_key", sa.String(length=255), nullable=False),
        sa.Column("state", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["workflow_run_id"], ["workflow_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_workflow_checkpoints_checkpoint_key", "workflow_checkpoints", ["checkpoint_key"])
    op.create_index("ix_workflow_checkpoints_workflow_run_id", "workflow_checkpoints", ["workflow_run_id"])

    op.create_table(
        "tool_call_logs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("workflow_run_id", sa.String(length=36), nullable=False),
        sa.Column("tool_name", sa.String(length=128), nullable=False),
        sa.Column("tool_group", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("input_payload", sa.JSON(), nullable=True),
        sa.Column("output_payload", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["workflow_run_id"], ["workflow_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_tool_call_logs_status", "tool_call_logs", ["status"])
    op.create_index("ix_tool_call_logs_tool_name", "tool_call_logs", ["tool_name"])
    op.create_index("ix_tool_call_logs_workflow_run_id", "tool_call_logs", ["workflow_run_id"])

    op.create_table(
        "approval_requests",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("workflow_run_id", sa.String(length=36), nullable=False),
        sa.Column("application_id", sa.String(length=36), nullable=True),
        sa.Column("action_type", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("decision", sa.String(length=64), nullable=True),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["application_id"], ["applications.id"]),
        sa.ForeignKeyConstraint(["workflow_run_id"], ["workflow_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_approval_requests_application_id", "approval_requests", ["application_id"])
    op.create_index("ix_approval_requests_status", "approval_requests", ["status"])
    op.create_index("ix_approval_requests_workflow_run_id", "approval_requests", ["workflow_run_id"])

    op.create_table(
        "automation_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("workflow_run_id", sa.String(length=36), nullable=False),
        sa.Column("application_id", sa.String(length=36), nullable=True),
        sa.Column("browser_session_id", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("target_url", sa.String(length=2048), nullable=True),
        sa.Column("last_screenshot_path", sa.String(length=1024), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["application_id"], ["applications.id"]),
        sa.ForeignKeyConstraint(["workflow_run_id"], ["workflow_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_automation_runs_application_id", "automation_runs", ["application_id"])
    op.create_index("ix_automation_runs_status", "automation_runs", ["status"])
    op.create_index("ix_automation_runs_workflow_run_id", "automation_runs", ["workflow_run_id"])


def downgrade() -> None:
    op.drop_index("ix_automation_runs_workflow_run_id", table_name="automation_runs")
    op.drop_index("ix_automation_runs_status", table_name="automation_runs")
    op.drop_index("ix_automation_runs_application_id", table_name="automation_runs")
    op.drop_table("automation_runs")
    op.drop_index("ix_approval_requests_workflow_run_id", table_name="approval_requests")
    op.drop_index("ix_approval_requests_status", table_name="approval_requests")
    op.drop_index("ix_approval_requests_application_id", table_name="approval_requests")
    op.drop_table("approval_requests")
    op.drop_index("ix_tool_call_logs_workflow_run_id", table_name="tool_call_logs")
    op.drop_index("ix_tool_call_logs_tool_name", table_name="tool_call_logs")
    op.drop_index("ix_tool_call_logs_status", table_name="tool_call_logs")
    op.drop_table("tool_call_logs")
    op.drop_index("ix_workflow_checkpoints_workflow_run_id", table_name="workflow_checkpoints")
    op.drop_index("ix_workflow_checkpoints_checkpoint_key", table_name="workflow_checkpoints")
    op.drop_table("workflow_checkpoints")
    op.drop_index("ix_workflow_runs_workflow_type", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_status", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_related_job_id", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_related_application_id", table_name="workflow_runs")
    op.drop_table("workflow_runs")
