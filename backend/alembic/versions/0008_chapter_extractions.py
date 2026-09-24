"""stored chapter extractions

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-25 10:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chapter_extractions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("book_id", sa.Uuid(), nullable=False),
        sa.Column("chapter_id", sa.Uuid(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column(
            "result",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["book_id"], ["books.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chapter_id"], ["chapters.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chapter_id"),
    )
    op.create_index(
        op.f("ix_chapter_extractions_book_id"), "chapter_extractions", ["book_id"], unique=False
    )
    op.create_index(
        op.f("ix_chapter_extractions_user_id"), "chapter_extractions", ["user_id"], unique=False
    )
    # Chapters extracted before this revision have nothing stored: their next
    # recomputation asks the model once (usually a cache hit) and stores the result.


def downgrade() -> None:
    op.drop_index(op.f("ix_chapter_extractions_user_id"), table_name="chapter_extractions")
    op.drop_index(op.f("ix_chapter_extractions_book_id"), table_name="chapter_extractions")
    op.drop_table("chapter_extractions")
