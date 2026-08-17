"""create agent skill metadata and usage tables

Revision ID: 20260814_0008
Revises: 20260814_0007
Create Date: 2026-08-14 00:00:00
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260814_0008"
down_revision: str | None = "20260814_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_skills",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description", sa.String(length=512), nullable=False),
        sa.Column("category", sa.String(length=128), nullable=False),
        sa.Column("storage_type", sa.String(length=32), nullable=False),
        sa.Column("file_path", sa.String(length=1024), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("protected", sa.Boolean(), nullable=False),
        sa.Column("pinned", sa.Boolean(), nullable=False),
        sa.Column("created_by", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_agent_skills_name"),
    )
    op.create_index("ix_agent_skills_category", "agent_skills", ["category"])
    op.create_index("ix_agent_skills_created_at", "agent_skills", ["created_at"])
    op.create_index("ix_agent_skills_status", "agent_skills", ["status"])

    op.create_table(
        "agent_skill_usage",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("skill_id", sa.String(length=36), nullable=False),
        sa.Column("use_count", sa.Integer(), nullable=False),
        sa.Column("view_count", sa.Integer(), nullable=False),
        sa.Column("patch_count", sa.Integer(), nullable=False),
        sa.Column("success_count", sa.Integer(), nullable=False),
        sa.Column("failure_count", sa.Integer(), nullable=False),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.Column("last_viewed_at", sa.DateTime(), nullable=True),
        sa.Column("last_patched_at", sa.DateTime(), nullable=True),
        sa.Column("last_success_at", sa.DateTime(), nullable=True),
        sa.Column("last_failure_at", sa.DateTime(), nullable=True),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("archived_at", sa.DateTime(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["skill_id"], ["agent_skills.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("skill_id", name="uq_agent_skill_usage_skill_id"),
    )
    op.create_index("ix_agent_skill_usage_skill_id", "agent_skill_usage", ["skill_id"])
    op.create_index("ix_agent_skill_usage_state", "agent_skill_usage", ["state"])


def downgrade() -> None:
    op.drop_index("ix_agent_skill_usage_state", table_name="agent_skill_usage")
    op.drop_index("ix_agent_skill_usage_skill_id", table_name="agent_skill_usage")
    op.drop_table("agent_skill_usage")
    op.drop_index("ix_agent_skills_status", table_name="agent_skills")
    op.drop_index("ix_agent_skills_created_at", table_name="agent_skills")
    op.drop_index("ix_agent_skills_category", table_name="agent_skills")
    op.drop_table("agent_skills")
