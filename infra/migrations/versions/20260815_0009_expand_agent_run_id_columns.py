"""expand agent run identifier columns

Revision ID: 20260815_0009
Revises: 20260814_0008
Create Date: 2026-08-15 00:00:00
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260815_0009"
down_revision: str | None = "20260814_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("agent_sessions") as batch_op:
        batch_op.alter_column(
            "current_agent_run_id",
            existing_type=sa.String(length=36),
            type_=sa.String(length=64),
            existing_nullable=True,
        )

    with op.batch_alter_table("agent_messages") as batch_op:
        batch_op.alter_column(
            "agent_run_id",
            existing_type=sa.String(length=36),
            type_=sa.String(length=64),
            existing_nullable=True,
        )

    with op.batch_alter_table("agent_learning_candidates") as batch_op:
        batch_op.alter_column(
            "source_agent_run_id",
            existing_type=sa.String(length=36),
            type_=sa.String(length=64),
            existing_nullable=True,
        )


def downgrade() -> None:
    with op.batch_alter_table("agent_learning_candidates") as batch_op:
        batch_op.alter_column(
            "source_agent_run_id",
            existing_type=sa.String(length=64),
            type_=sa.String(length=36),
            existing_nullable=True,
        )

    with op.batch_alter_table("agent_messages") as batch_op:
        batch_op.alter_column(
            "agent_run_id",
            existing_type=sa.String(length=64),
            type_=sa.String(length=36),
            existing_nullable=True,
        )

    with op.batch_alter_table("agent_sessions") as batch_op:
        batch_op.alter_column(
            "current_agent_run_id",
            existing_type=sa.String(length=64),
            type_=sa.String(length=36),
            existing_nullable=True,
        )
