"""create WeChat article candidates and recruiting signals

Revision ID: 20260812_0005
Revises: 20260812_0004
Create Date: 2026-08-12 00:00:00
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260812_0005"
down_revision: str | None = "20260812_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "article_candidates",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source_id", sa.String(length=36), nullable=False),
        sa.Column("sync_run_id", sa.String(length=36), nullable=True),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("url", sa.String(length=2048), nullable=False),
        sa.Column("url_hash", sa.String(length=64), nullable=False),
        sa.Column("source_account", sa.String(length=255), nullable=True),
        sa.Column("published_at", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("raw_payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["source_id"], ["job_sources.id"]),
        sa.ForeignKeyConstraint(["sync_run_id"], ["source_sync_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_id", "url_hash", name="uq_article_candidates_source_url_hash"),
    )
    op.create_index("ix_article_candidates_source_account", "article_candidates", ["source_account"])
    op.create_index("ix_article_candidates_source_id", "article_candidates", ["source_id"])
    op.create_index("ix_article_candidates_status", "article_candidates", ["status"])
    op.create_index("ix_article_candidates_sync_run_id", "article_candidates", ["sync_run_id"])

    op.create_table(
        "recruiting_signals",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source_id", sa.String(length=36), nullable=False),
        sa.Column("raw_lead_id", sa.String(length=36), nullable=True),
        sa.Column("article_candidate_id", sa.String(length=36), nullable=True),
        sa.Column("signal_hash", sa.String(length=64), nullable=False),
        sa.Column("company_name", sa.String(length=255), nullable=False),
        sa.Column("normalized_company_name", sa.String(length=255), nullable=False),
        sa.Column("signal_type", sa.String(length=64), nullable=False),
        sa.Column("graduation_year", sa.String(length=32), nullable=True),
        sa.Column("source_url", sa.String(length=2048), nullable=True),
        sa.Column("original_source", sa.String(length=255), nullable=True),
        sa.Column("confidence_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("trust_level", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("raw_payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["article_candidate_id"], ["article_candidates.id"]),
        sa.ForeignKeyConstraint(["raw_lead_id"], ["raw_job_leads.id"]),
        sa.ForeignKeyConstraint(["source_id"], ["job_sources.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_id", "signal_hash", name="uq_recruiting_signals_source_hash"),
    )
    op.create_index("ix_recruiting_signals_article_candidate_id", "recruiting_signals", ["article_candidate_id"])
    op.create_index("ix_recruiting_signals_company_name", "recruiting_signals", ["company_name"])
    op.create_index("ix_recruiting_signals_raw_lead_id", "recruiting_signals", ["raw_lead_id"])
    op.create_index("ix_recruiting_signals_source_id", "recruiting_signals", ["source_id"])
    op.create_index("ix_recruiting_signals_status", "recruiting_signals", ["status"])


def downgrade() -> None:
    op.drop_index("ix_recruiting_signals_status", table_name="recruiting_signals")
    op.drop_index("ix_recruiting_signals_source_id", table_name="recruiting_signals")
    op.drop_index("ix_recruiting_signals_raw_lead_id", table_name="recruiting_signals")
    op.drop_index("ix_recruiting_signals_company_name", table_name="recruiting_signals")
    op.drop_index("ix_recruiting_signals_article_candidate_id", table_name="recruiting_signals")
    op.drop_table("recruiting_signals")
    op.drop_index("ix_article_candidates_sync_run_id", table_name="article_candidates")
    op.drop_index("ix_article_candidates_status", table_name="article_candidates")
    op.drop_index("ix_article_candidates_source_id", table_name="article_candidates")
    op.drop_index("ix_article_candidates_source_account", table_name="article_candidates")
    op.drop_table("article_candidates")
