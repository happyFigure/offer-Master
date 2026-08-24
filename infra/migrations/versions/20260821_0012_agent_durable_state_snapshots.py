"""create agent durable memory snapshot and artifact tables

Revision ID: 20260821_0012
Revises: 20260821_0011
Create Date: 2026-08-21 00:12:00
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260821_0012"
down_revision: str | None = "20260821_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_memory_snapshots",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("step_id", sa.String(length=64), nullable=True),
        sa.Column("memory_id", sa.String(length=128), nullable=False),
        sa.Column("source_type", sa.String(length=128), nullable=False),
        sa.Column("usage_reason", sa.String(length=1024), nullable=False),
        sa.Column("visibility_scope", sa.String(length=64), nullable=False),
        sa.Column("passed_to_executor", sa.Boolean(), nullable=False),
        sa.Column("memory_payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["step_id"], ["agent_step_states.id"]),
        sa.ForeignKeyConstraint(["task_id"], ["agent_task_states.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_memory_snapshots_memory_id", "agent_memory_snapshots", ["memory_id"])
    op.create_index("ix_agent_memory_snapshots_passed_to_executor", "agent_memory_snapshots", ["passed_to_executor"])
    op.create_index("ix_agent_memory_snapshots_step_id", "agent_memory_snapshots", ["step_id"])
    op.create_index("ix_agent_memory_snapshots_task_id", "agent_memory_snapshots", ["task_id"])
    op.create_index("ix_agent_memory_snapshots_visibility_scope", "agent_memory_snapshots", ["visibility_scope"])

    op.create_table(
        "agent_artifact_index",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("step_id", sa.String(length=64), nullable=True),
        sa.Column("sequence_index", sa.Integer(), nullable=False),
        sa.Column("source_kind", sa.String(length=64), nullable=False),
        sa.Column("artifact_type", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=True),
        sa.Column("uri", sa.String(length=2048), nullable=False),
        sa.Column("mime_type", sa.String(length=128), nullable=True),
        sa.Column("artifact_metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["step_id"], ["agent_step_states.id"]),
        sa.ForeignKeyConstraint(["task_id"], ["agent_task_states.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_artifact_index_artifact_type", "agent_artifact_index", ["artifact_type"])
    op.create_index("ix_agent_artifact_index_sequence_index", "agent_artifact_index", ["sequence_index"])
    op.create_index("ix_agent_artifact_index_source_kind", "agent_artifact_index", ["source_kind"])
    op.create_index("ix_agent_artifact_index_step_id", "agent_artifact_index", ["step_id"])
    op.create_index("ix_agent_artifact_index_task_id", "agent_artifact_index", ["task_id"])


def downgrade() -> None:
    op.drop_index("ix_agent_artifact_index_task_id", table_name="agent_artifact_index")
    op.drop_index("ix_agent_artifact_index_step_id", table_name="agent_artifact_index")
    op.drop_index("ix_agent_artifact_index_source_kind", table_name="agent_artifact_index")
    op.drop_index("ix_agent_artifact_index_sequence_index", table_name="agent_artifact_index")
    op.drop_index("ix_agent_artifact_index_artifact_type", table_name="agent_artifact_index")
    op.drop_table("agent_artifact_index")
    op.drop_index("ix_agent_memory_snapshots_visibility_scope", table_name="agent_memory_snapshots")
    op.drop_index("ix_agent_memory_snapshots_task_id", table_name="agent_memory_snapshots")
    op.drop_index("ix_agent_memory_snapshots_step_id", table_name="agent_memory_snapshots")
    op.drop_index("ix_agent_memory_snapshots_passed_to_executor", table_name="agent_memory_snapshots")
    op.drop_index("ix_agent_memory_snapshots_memory_id", table_name="agent_memory_snapshots")
    op.drop_table("agent_memory_snapshots")
