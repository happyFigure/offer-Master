"""add structured QQ Mail notification event fields

Revision ID: 20261001_0013
Revises: 20260821_0012
Create Date: 2026-10-01 00:13:00
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20261001_0013"
down_revision: str | None = "20260821_0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # MySQL can add these nullable fields and indexes in place.  Forcing a
    # table rebuild makes Alembic copy the existing auto-named foreign key
    # (`application_events_ibfk_1`) onto a temporary table, which MySQL
    # rejects because foreign-key names are schema-wide.
    with op.batch_alter_table("application_events") as batch_op:
        batch_op.add_column(sa.Column("scheduled_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("deadline_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("timezone", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("join_url", sa.String(length=2048), nullable=True))
        batch_op.add_column(sa.Column("source_message_id", sa.String(length=512), nullable=True))
        batch_op.add_column(sa.Column("source_uid", sa.String(length=128), nullable=True))
        batch_op.add_column(sa.Column("review_status", sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column("reviewed_at", sa.DateTime(), nullable=True))
        batch_op.create_index("ix_application_events_review_status", ["review_status"])
        batch_op.create_unique_constraint(
            "uq_application_events_mail_source",
            ["application_id", "source_message_id", "event_type"],
        )


def downgrade() -> None:
    with op.batch_alter_table("application_events") as batch_op:
        batch_op.drop_constraint("uq_application_events_mail_source", type_="unique")
        batch_op.drop_index("ix_application_events_review_status")
        batch_op.drop_column("reviewed_at")
        batch_op.drop_column("review_status")
        batch_op.drop_column("source_uid")
        batch_op.drop_column("source_message_id")
        batch_op.drop_column("join_url")
        batch_op.drop_column("timezone")
        batch_op.drop_column("deadline_at")
        batch_op.drop_column("scheduled_at")
