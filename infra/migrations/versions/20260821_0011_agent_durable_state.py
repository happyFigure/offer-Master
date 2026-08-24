"""create agent durable state tables

Revision ID: 20260821_0011
Revises: 20260818_0010
Create Date: 2026-08-21 00:11:00
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260821_0011"
down_revision: str | None = "20260818_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_task_states",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("root_workflow_run_id", sa.String(length=64), nullable=False),
        sa.Column("conversation_session_id", sa.String(length=64), nullable=False),
        sa.Column("task_type", sa.String(length=128), nullable=False),
        sa.Column("capability", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("current_step_id", sa.String(length=64), nullable=True),
        sa.Column("owner_executor", sa.String(length=128), nullable=True),
        sa.Column("user_goal", sa.String(length=2048), nullable=True),
        sa.Column("input_payload", sa.JSON(), nullable=False),
        sa.Column("output_payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_task_states_capability", "agent_task_states", ["capability"])
    op.create_index("ix_agent_task_states_conversation_session_id", "agent_task_states", ["conversation_session_id"])
    op.create_index("ix_agent_task_states_root_workflow_run_id", "agent_task_states", ["root_workflow_run_id"])
    op.create_index("ix_agent_task_states_status", "agent_task_states", ["status"])
    op.create_index("ix_agent_task_states_task_type", "agent_task_states", ["task_type"])

    op.create_table(
        "agent_step_states",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("parent_step_id", sa.String(length=64), nullable=True),
        sa.Column("sequence_index", sa.Integer(), nullable=False),
        sa.Column("step_type", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("executor_type", sa.String(length=128), nullable=False),
        sa.Column("executor_name", sa.String(length=128), nullable=False),
        sa.Column("capability", sa.String(length=128), nullable=False),
        sa.Column("input_payload", sa.JSON(), nullable=False),
        sa.Column("output_payload", sa.JSON(), nullable=False),
        sa.Column("tool_call_log_id", sa.String(length=64), nullable=True),
        sa.Column("external_task_id", sa.String(length=64), nullable=True),
        sa.Column("approval_request_id", sa.String(length=64), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["task_id"], ["agent_task_states.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_step_states_approval_request_id", "agent_step_states", ["approval_request_id"])
    op.create_index("ix_agent_step_states_capability", "agent_step_states", ["capability"])
    op.create_index("ix_agent_step_states_external_task_id", "agent_step_states", ["external_task_id"])
    op.create_index("ix_agent_step_states_status", "agent_step_states", ["status"])
    op.create_index("ix_agent_step_states_task_id", "agent_step_states", ["task_id"])
    op.create_index("ix_agent_step_states_tool_call_log_id", "agent_step_states", ["tool_call_log_id"])


def downgrade() -> None:
    op.drop_index("ix_agent_step_states_tool_call_log_id", table_name="agent_step_states")
    op.drop_index("ix_agent_step_states_task_id", table_name="agent_step_states")
    op.drop_index("ix_agent_step_states_status", table_name="agent_step_states")
    op.drop_index("ix_agent_step_states_external_task_id", table_name="agent_step_states")
    op.drop_index("ix_agent_step_states_capability", table_name="agent_step_states")
    op.drop_index("ix_agent_step_states_approval_request_id", table_name="agent_step_states")
    op.drop_table("agent_step_states")
    op.drop_index("ix_agent_task_states_task_type", table_name="agent_task_states")
    op.drop_index("ix_agent_task_states_status", table_name="agent_task_states")
    op.drop_index("ix_agent_task_states_root_workflow_run_id", table_name="agent_task_states")
    op.drop_index("ix_agent_task_states_conversation_session_id", table_name="agent_task_states")
    op.drop_index("ix_agent_task_states_capability", table_name="agent_task_states")
    op.drop_table("agent_task_states")
