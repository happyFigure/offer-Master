"""create external agent task tables

Revision ID: 20260818_0010
Revises: 20260815_0009
Create Date: 2026-08-18 00:10:00
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260818_0010"
down_revision: str | None = "20260815_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "external_agent_tasks",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("task_type", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("trace_id", sa.String(length=128), nullable=True),
        sa.Column("context_pack_hash", sa.String(length=64), nullable=False),
        sa.Column("input_payload", sa.JSON(), nullable=False),
        sa.Column("output_payload", sa.JSON(), nullable=True),
        sa.Column("blocked_reason", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_external_agent_tasks_status", "external_agent_tasks", ["status"])
    op.create_index("ix_external_agent_tasks_task_type", "external_agent_tasks", ["task_type"])
    op.create_index("ix_external_agent_tasks_trace_id", "external_agent_tasks", ["trace_id"])
    op.create_index("ix_external_agent_tasks_updated_at", "external_agent_tasks", ["updated_at"])

    op.create_table(
        "external_agent_task_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("event_type", sa.String(length=128), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["task_id"], ["external_agent_tasks.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_external_agent_task_events_created_at", "external_agent_task_events", ["created_at"])
    op.create_index("ix_external_agent_task_events_event_type", "external_agent_task_events", ["event_type"])
    op.create_index("ix_external_agent_task_events_task_id", "external_agent_task_events", ["task_id"])

    op.create_table(
        "external_agent_artifacts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("artifact_type", sa.String(length=128), nullable=False),
        sa.Column("path_or_uri", sa.String(length=2048), nullable=False),
        sa.Column("mime_type", sa.String(length=128), nullable=True),
        sa.Column("artifact_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["task_id"], ["external_agent_tasks.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_external_agent_artifacts_artifact_type", "external_agent_artifacts", ["artifact_type"])
    op.create_index("ix_external_agent_artifacts_created_at", "external_agent_artifacts", ["created_at"])
    op.create_index("ix_external_agent_artifacts_task_id", "external_agent_artifacts", ["task_id"])


def downgrade() -> None:
    op.drop_index("ix_external_agent_artifacts_task_id", table_name="external_agent_artifacts")
    op.drop_index("ix_external_agent_artifacts_created_at", table_name="external_agent_artifacts")
    op.drop_index("ix_external_agent_artifacts_artifact_type", table_name="external_agent_artifacts")
    op.drop_table("external_agent_artifacts")
    op.drop_index("ix_external_agent_task_events_task_id", table_name="external_agent_task_events")
    op.drop_index("ix_external_agent_task_events_event_type", table_name="external_agent_task_events")
    op.drop_index("ix_external_agent_task_events_created_at", table_name="external_agent_task_events")
    op.drop_table("external_agent_task_events")
    op.drop_index("ix_external_agent_tasks_updated_at", table_name="external_agent_tasks")
    op.drop_index("ix_external_agent_tasks_trace_id", table_name="external_agent_tasks")
    op.drop_index("ix_external_agent_tasks_task_type", table_name="external_agent_tasks")
    op.drop_index("ix_external_agent_tasks_status", table_name="external_agent_tasks")
    op.drop_table("external_agent_tasks")
