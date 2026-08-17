"""create agent memory and learning candidate tables

Revision ID: 20260814_0007
Revises: 20260814_0006
Create Date: 2026-08-14 00:00:00
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260814_0007"
down_revision: str | None = "20260814_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_memories",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("memory_type", sa.String(length=64), nullable=False),
        sa.Column("scope", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=128), nullable=True),
        sa.Column("source_id", sa.String(length=36), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("importance", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_memories_created_at", "agent_memories", ["created_at"])
    op.create_index("ix_agent_memories_scope", "agent_memories", ["scope"])
    op.create_index("ix_agent_memories_source_type", "agent_memories", ["source_type"])
    op.create_index("ix_agent_memories_status", "agent_memories", ["status"])

    op.create_table(
        "agent_learning_candidates",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source_agent_run_id", sa.String(length=36), nullable=True),
        sa.Column("source_workflow_run_id", sa.String(length=36), nullable=True),
        sa.Column("source_tool_call_log_id", sa.String(length=36), nullable=True),
        sa.Column("source_message_id", sa.String(length=36), nullable=True),
        sa.Column("lesson_type", sa.String(length=64), nullable=False),
        sa.Column("target_scope", sa.String(length=128), nullable=False),
        sa.Column("suggested_skill_target", sa.String(length=128), nullable=False),
        sa.Column("target_skill_id", sa.String(length=36), nullable=True),
        sa.Column("candidate_title", sa.String(length=255), nullable=False),
        sa.Column("candidate_body", sa.Text(), nullable=False),
        sa.Column("evidence_summary", sa.Text(), nullable=False),
        sa.Column("success_evidence", sa.Text(), nullable=True),
        sa.Column("evidence_json", sa.JSON(), nullable=True),
        sa.Column("risk_level", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("reviewed_by", sa.String(length=64), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("applied_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_learning_candidates_created_at", "agent_learning_candidates", ["created_at"])
    op.create_index("ix_agent_learning_candidates_source_tool_call_log_id", "agent_learning_candidates", ["source_tool_call_log_id"])
    op.create_index("ix_agent_learning_candidates_source_workflow_run_id", "agent_learning_candidates", ["source_workflow_run_id"])
    op.create_index("ix_agent_learning_candidates_status", "agent_learning_candidates", ["status"])
    op.create_index("ix_agent_learning_candidates_target_scope", "agent_learning_candidates", ["target_scope"])


def downgrade() -> None:
    op.drop_index("ix_agent_learning_candidates_target_scope", table_name="agent_learning_candidates")
    op.drop_index("ix_agent_learning_candidates_status", table_name="agent_learning_candidates")
    op.drop_index("ix_agent_learning_candidates_source_workflow_run_id", table_name="agent_learning_candidates")
    op.drop_index("ix_agent_learning_candidates_source_tool_call_log_id", table_name="agent_learning_candidates")
    op.drop_index("ix_agent_learning_candidates_created_at", table_name="agent_learning_candidates")
    op.drop_table("agent_learning_candidates")
    op.drop_index("ix_agent_memories_status", table_name="agent_memories")
    op.drop_index("ix_agent_memories_source_type", table_name="agent_memories")
    op.drop_index("ix_agent_memories_scope", table_name="agent_memories")
    op.drop_index("ix_agent_memories_created_at", table_name="agent_memories")
    op.drop_table("agent_memories")
