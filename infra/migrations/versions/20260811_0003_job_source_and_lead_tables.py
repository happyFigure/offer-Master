"""create job source and lead tables

Revision ID: 20260811_0003
Revises: 20260810_0002
Create Date: 2026-08-11 00:00:00
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260811_0003"
down_revision: str | None = "20260810_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "job_sources",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("entry_url", sa.String(length=2048), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("sync_interval_hours", sa.Integer(), nullable=False),
        sa.Column("trust_level", sa.String(length=32), nullable=False),
        sa.Column("fetch_mode", sa.String(length=64), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("raw_payload", sa.JSON(), nullable=True),
        sa.Column("last_synced_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_job_sources_enabled", "job_sources", ["enabled"])
    op.create_index("ix_job_sources_source_type", "job_sources", ["source_type"])

    op.create_table(
        "source_sync_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("fetched_count", sa.Integer(), nullable=False),
        sa.Column("extracted_count", sa.Integer(), nullable=False),
        sa.Column("failed_count", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("run_metadata", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["source_id"], ["job_sources.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_source_sync_runs_source_id", "source_sync_runs", ["source_id"])
    op.create_index("ix_source_sync_runs_started_at", "source_sync_runs", ["started_at"])
    op.create_index("ix_source_sync_runs_status", "source_sync_runs", ["status"])

    op.create_table(
        "raw_job_leads",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source_id", sa.String(length=36), nullable=False),
        sa.Column("sync_run_id", sa.String(length=36), nullable=True),
        sa.Column("source_url", sa.String(length=2048), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("content_type", sa.String(length=128), nullable=False),
        sa.Column("raw_content", sa.Text(), nullable=False),
        sa.Column("extracted_text", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("raw_payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["source_id"], ["job_sources.id"]),
        sa.ForeignKeyConstraint(["sync_run_id"], ["source_sync_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_id", "content_hash", name="uq_raw_job_leads_source_hash"),
    )
    op.create_index("ix_raw_job_leads_source_id", "raw_job_leads", ["source_id"])
    op.create_index("ix_raw_job_leads_status", "raw_job_leads", ["status"])
    op.create_index("ix_raw_job_leads_sync_run_id", "raw_job_leads", ["sync_run_id"])

    op.create_table(
        "job_leads",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source_id", sa.String(length=36), nullable=False),
        sa.Column("raw_lead_id", sa.String(length=36), nullable=True),
        sa.Column("converted_job_id", sa.String(length=36), nullable=True),
        sa.Column("lead_hash", sa.String(length=64), nullable=False),
        sa.Column("company_name", sa.String(length=255), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("city", sa.String(length=128), nullable=True),
        sa.Column("job_direction", sa.String(length=128), nullable=True),
        sa.Column("graduation_year", sa.String(length=32), nullable=True),
        sa.Column("source_url", sa.String(length=2048), nullable=True),
        sa.Column("apply_url", sa.String(length=2048), nullable=True),
        sa.Column("verified_url", sa.String(length=2048), nullable=True),
        sa.Column("job_type", sa.String(length=128), nullable=True),
        sa.Column("salary_text", sa.String(length=255), nullable=True),
        sa.Column("jd_text", sa.Text(), nullable=True),
        sa.Column("skills", sa.JSON(), nullable=False),
        sa.Column("deadline", sa.Date(), nullable=True),
        sa.Column("confidence_score", sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column("trust_level", sa.String(length=32), nullable=False),
        sa.Column("verification_status", sa.String(length=32), nullable=False),
        sa.Column("verification_notes", sa.Text(), nullable=True),
        sa.Column("verified_at", sa.DateTime(), nullable=True),
        sa.Column("converted_at", sa.DateTime(), nullable=True),
        sa.Column("raw_payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["converted_job_id"], ["jobs.id"]),
        sa.ForeignKeyConstraint(["raw_lead_id"], ["raw_job_leads.id"]),
        sa.ForeignKeyConstraint(["source_id"], ["job_sources.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_id", "lead_hash", name="uq_job_leads_source_hash"),
    )
    op.create_index("ix_job_leads_company_name", "job_leads", ["company_name"])
    op.create_index("ix_job_leads_converted_job_id", "job_leads", ["converted_job_id"])
    op.create_index("ix_job_leads_raw_lead_id", "job_leads", ["raw_lead_id"])
    op.create_index("ix_job_leads_source_id", "job_leads", ["source_id"])
    op.create_index(
        "ix_job_leads_verification_status",
        "job_leads",
        ["verification_status"],
    )


def downgrade() -> None:
    op.drop_index("ix_job_leads_verification_status", table_name="job_leads")
    op.drop_index("ix_job_leads_source_id", table_name="job_leads")
    op.drop_index("ix_job_leads_raw_lead_id", table_name="job_leads")
    op.drop_index("ix_job_leads_converted_job_id", table_name="job_leads")
    op.drop_index("ix_job_leads_company_name", table_name="job_leads")
    op.drop_table("job_leads")
    op.drop_index("ix_raw_job_leads_sync_run_id", table_name="raw_job_leads")
    op.drop_index("ix_raw_job_leads_status", table_name="raw_job_leads")
    op.drop_index("ix_raw_job_leads_source_id", table_name="raw_job_leads")
    op.drop_table("raw_job_leads")
    op.drop_index("ix_source_sync_runs_status", table_name="source_sync_runs")
    op.drop_index("ix_source_sync_runs_started_at", table_name="source_sync_runs")
    op.drop_index("ix_source_sync_runs_source_id", table_name="source_sync_runs")
    op.drop_table("source_sync_runs")
    op.drop_index("ix_job_sources_source_type", table_name="job_sources")
    op.drop_index("ix_job_sources_enabled", table_name="job_sources")
    op.drop_table("job_sources")
