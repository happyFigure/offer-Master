"""create agent conversation memory tables

Revision ID: 20260814_0006
Revises: 20260812_0005
Create Date: 2026-08-14 00:00:00
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260814_0006"
down_revision: str | None = "20260812_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_sessions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("primary_intent", sa.String(length=128), nullable=True),
        sa.Column("current_agent_run_id", sa.String(length=36), nullable=True),
        sa.Column("last_context_summary_id", sa.String(length=36), nullable=True),
        sa.Column("message_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("last_message_at", sa.DateTime(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_sessions_last_message_at", "agent_sessions", ["last_message_at"])
    op.create_index("ix_agent_sessions_primary_intent", "agent_sessions", ["primary_intent"])
    op.create_index("ix_agent_sessions_status", "agent_sessions", ["status"])

    op.create_table(
        "agent_context_summaries",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("summary_json", sa.JSON(), nullable=True),
        sa.Column("covered_message_start_id", sa.String(length=36), nullable=True),
        sa.Column("covered_message_end_id", sa.String(length=36), nullable=True),
        sa.Column("first_kept_message_id", sa.String(length=36), nullable=True),
        sa.Column("previous_summary_id", sa.String(length=36), nullable=True),
        sa.Column("token_estimate", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("created_by", sa.String(length=128), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["previous_summary_id"], ["agent_context_summaries.id"]),
        sa.ForeignKeyConstraint(["session_id"], ["agent_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_context_summaries_created_at", "agent_context_summaries", ["created_at"])
    op.create_index(
        "ix_agent_context_summaries_previous_summary_id",
        "agent_context_summaries",
        ["previous_summary_id"],
    )
    op.create_index("ix_agent_context_summaries_session_id", "agent_context_summaries", ["session_id"])

    op.create_table(
        "agent_messages",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("message_kind", sa.String(length=64), nullable=False),
        sa.Column("agent_id", sa.String(length=128), nullable=True),
        sa.Column("recipient_agent_id", sa.String(length=128), nullable=True),
        sa.Column("visibility_scope", sa.String(length=32), nullable=False),
        sa.Column("content_text", sa.Text(), nullable=True),
        sa.Column("content_json", sa.JSON(), nullable=True),
        sa.Column("visible_content_text", sa.Text(), nullable=True),
        sa.Column("runtime_content_text", sa.Text(), nullable=True),
        sa.Column("content_type", sa.String(length=128), nullable=False),
        sa.Column("provenance_kind", sa.String(length=64), nullable=True),
        sa.Column("agent_run_id", sa.String(length=36), nullable=True),
        sa.Column("workflow_run_id", sa.String(length=36), nullable=True),
        sa.Column("tool_call_log_id", sa.String(length=36), nullable=True),
        sa.Column("parent_message_id", sa.String(length=36), nullable=True),
        sa.Column("token_estimate", sa.Integer(), nullable=True),
        sa.Column("exclude_from_context", sa.Boolean(), nullable=False),
        sa.Column("compacted_by_summary_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["compacted_by_summary_id"], ["agent_context_summaries.id"]),
        sa.ForeignKeyConstraint(["parent_message_id"], ["agent_messages.id"]),
        sa.ForeignKeyConstraint(["session_id"], ["agent_sessions.id"]),
        sa.ForeignKeyConstraint(["tool_call_log_id"], ["tool_call_logs.id"]),
        sa.ForeignKeyConstraint(["workflow_run_id"], ["workflow_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_messages_agent_run_id", "agent_messages", ["agent_run_id"])
    op.create_index("ix_agent_messages_compacted_by_summary_id", "agent_messages", ["compacted_by_summary_id"])
    op.create_index("ix_agent_messages_created_at", "agent_messages", ["created_at"])
    op.create_index("ix_agent_messages_message_kind", "agent_messages", ["message_kind"])
    op.create_index("ix_agent_messages_role", "agent_messages", ["role"])
    op.create_index("ix_agent_messages_session_id", "agent_messages", ["session_id"])
    op.create_index("ix_agent_messages_tool_call_log_id", "agent_messages", ["tool_call_log_id"])
    op.create_index("ix_agent_messages_workflow_run_id", "agent_messages", ["workflow_run_id"])


def downgrade() -> None:
    op.drop_index("ix_agent_messages_workflow_run_id", table_name="agent_messages")
    op.drop_index("ix_agent_messages_tool_call_log_id", table_name="agent_messages")
    op.drop_index("ix_agent_messages_session_id", table_name="agent_messages")
    op.drop_index("ix_agent_messages_role", table_name="agent_messages")
    op.drop_index("ix_agent_messages_message_kind", table_name="agent_messages")
    op.drop_index("ix_agent_messages_created_at", table_name="agent_messages")
    op.drop_index("ix_agent_messages_compacted_by_summary_id", table_name="agent_messages")
    op.drop_index("ix_agent_messages_agent_run_id", table_name="agent_messages")
    op.drop_table("agent_messages")
    op.drop_index("ix_agent_context_summaries_session_id", table_name="agent_context_summaries")
    op.drop_index("ix_agent_context_summaries_previous_summary_id", table_name="agent_context_summaries")
    op.drop_index("ix_agent_context_summaries_created_at", table_name="agent_context_summaries")
    op.drop_table("agent_context_summaries")
    op.drop_index("ix_agent_sessions_status", table_name="agent_sessions")
    op.drop_index("ix_agent_sessions_primary_intent", table_name="agent_sessions")
    op.drop_index("ix_agent_sessions_last_message_at", table_name="agent_sessions")
    op.drop_table("agent_sessions")
