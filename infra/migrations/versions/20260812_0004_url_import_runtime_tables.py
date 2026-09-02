"""create URL import runtime tables

Revision ID: 20260812_0004
Revises: 20260811_0003
Create Date: 2026-08-12 00:00:00
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260812_0004"
down_revision: str | None = "20260811_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "domain_health_states",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("domain", sa.String(length=255), nullable=False),
        sa.Column("tool_name", sa.String(length=128), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("failure_count", sa.Integer(), nullable=False),
        sa.Column("success_count", sa.Integer(), nullable=False),
        sa.Column("last_error_code", sa.String(length=64), nullable=True),
        sa.Column("last_error_message", sa.Text(), nullable=True),
        sa.Column("opened_at", sa.DateTime(), nullable=True),
        sa.Column("cooldown_until", sa.DateTime(), nullable=True),
        sa.Column("half_open_probe_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("domain", "tool_name", name="uq_domain_health_domain_tool"),
    )
    op.create_index("ix_domain_health_states_cooldown_until", "domain_health_states", ["cooldown_until"])
    op.create_index("ix_domain_health_states_domain", "domain_health_states", ["domain"])
    op.create_index("ix_domain_health_states_state", "domain_health_states", ["state"])

    op.create_table(
        "url_import_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("workflow_run_id", sa.String(length=36), nullable=False),
        sa.Column("source_id", sa.String(length=36), nullable=True),
        sa.Column("input_url", sa.String(length=2048), nullable=False),
        sa.Column("normalized_url", sa.String(length=2048), nullable=True),
        sa.Column("normalized_url_hash", sa.String(length=64), nullable=True),
        sa.Column("source_type", sa.String(length=64), nullable=True),
        sa.Column("domain", sa.String(length=255), nullable=True),
        sa.Column("fetch_layer", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("current_stage", sa.String(length=128), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("tool_call_count", sa.Integer(), nullable=False),
        sa.Column("llm_call_count", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("next_action", sa.String(length=128), nullable=True),
        sa.Column("raw_job_lead_id", sa.String(length=36), nullable=True),
        sa.Column("extracted_count", sa.Integer(), nullable=False),
        sa.Column("duplicate_of_run_id", sa.String(length=36), nullable=True),
        sa.Column("run_metadata", sa.JSON(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["duplicate_of_run_id"], ["url_import_runs.id"]),
        sa.ForeignKeyConstraint(["raw_job_lead_id"], ["raw_job_leads.id"]),
        sa.ForeignKeyConstraint(["source_id"], ["job_sources.id"]),
        sa.ForeignKeyConstraint(["workflow_run_id"], ["workflow_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_url_import_runs_current_stage", "url_import_runs", ["current_stage"])
    op.create_index("ix_url_import_runs_domain", "url_import_runs", ["domain"])
    op.create_index(
        "ix_url_import_runs_normalized_url_hash",
        "url_import_runs",
        ["normalized_url_hash"],
    )
    op.create_index("ix_url_import_runs_source_id", "url_import_runs", ["source_id"])
    op.create_index("ix_url_import_runs_status", "url_import_runs", ["status"])
    op.create_index("ix_url_import_runs_workflow_run_id", "url_import_runs", ["workflow_run_id"])


def downgrade() -> None:
    op.drop_index("ix_url_import_runs_workflow_run_id", table_name="url_import_runs")
    op.drop_index("ix_url_import_runs_status", table_name="url_import_runs")
    op.drop_index("ix_url_import_runs_source_id", table_name="url_import_runs")
    op.drop_index("ix_url_import_runs_normalized_url_hash", table_name="url_import_runs")
    op.drop_index("ix_url_import_runs_domain", table_name="url_import_runs")
    op.drop_index("ix_url_import_runs_current_stage", table_name="url_import_runs")
    op.drop_table("url_import_runs")
    op.drop_index("ix_domain_health_states_state", table_name="domain_health_states")
    op.drop_index("ix_domain_health_states_domain", table_name="domain_health_states")
    op.drop_index("ix_domain_health_states_cooldown_until", table_name="domain_health_states")
    op.drop_table("domain_health_states")
