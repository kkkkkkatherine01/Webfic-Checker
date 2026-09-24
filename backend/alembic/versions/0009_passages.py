"""passages for text search (pgvector)

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-25 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    postgres = op.get_bind().dialect.name == "postgresql"
    if postgres:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "passages",
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
        sa.Column("chapter_number", sa.Integer(), nullable=False),
        sa.Column("source_key", sa.String(length=64), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("tokens", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(512) if postgres else sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["book_id"], ["books.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chapter_id"], ["chapters.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_passages_user_id"), "passages", ["user_id"], unique=False)
    op.create_index(op.f("ix_passages_chapter_id"), "passages", ["chapter_id"], unique=False)
    op.create_index(
        "ix_passages_book_chapter", "passages", ["book_id", "chapter_number"], unique=False
    )
    if postgres:
        # Keyword search. Queries must use this exact expression to hit the index.
        # No ANN index on the embedding: searches are filtered to one book (and often a
        # chapter range or character), where an exact scan is fast and never misses.
        op.execute(
            "CREATE INDEX ix_passages_tokens ON passages "
            "USING gin (to_tsvector('simple'::regconfig, tokens))"
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_passages_tokens")
    op.drop_index("ix_passages_book_chapter", table_name="passages")
    op.drop_index(op.f("ix_passages_chapter_id"), table_name="passages")
    op.drop_index(op.f("ix_passages_user_id"), table_name="passages")
    op.drop_table("passages")
    # The vector extension is left installed: dropping it needs superuser rights and
    # nothing else depends on it.
