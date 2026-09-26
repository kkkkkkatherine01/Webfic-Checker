"""issue verifications

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-26 23:36:06.045997
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "issue_verifications",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("book_id", sa.Uuid(), nullable=False),
        sa.Column("issue_id", sa.Uuid(), nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("verdict", sa.String(length=16), nullable=True),
        sa.Column("reason", sa.String(length=16), nullable=True),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column(
            "evidence",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("prompt_version", sa.String(length=32), nullable=False),
        sa.Column("cost_usd", sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["book_id"], ["books.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["issue_id"], ["consistency_issues.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_issue_verifications_book_id"), "issue_verifications", ["book_id"], unique=False
    )
    op.create_index(
        op.f("ix_issue_verifications_issue_id"), "issue_verifications", ["issue_id"], unique=False
    )
    op.create_index(
        op.f("ix_issue_verifications_user_id"), "issue_verifications", ["user_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_issue_verifications_user_id"), table_name="issue_verifications")
    op.drop_index(op.f("ix_issue_verifications_issue_id"), table_name="issue_verifications")
    op.drop_index(op.f("ix_issue_verifications_book_id"), table_name="issue_verifications")
    op.drop_table("issue_verifications")
