"""extraction kinds: one stored extraction per chapter and kind

Revision ID: 0014
Revises: 0013
Create Date: 2026-09-27

Step 5-0: extraction becomes several kinds (ages first, character facts next), each
stored and reused on its own. Existing rows are all age extractions.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _table(kind: bool) -> sa.Table:
    """The table without its unique constraint, with or without the kind column: SQLite
    rebuilds a table to change constraints, and the old one has no name to drop it by,
    so the rebuild starts from this definition and adds the wanted constraint."""
    columns = [
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
        sa.Index("ix_chapter_extractions_book_id", "book_id"),
        sa.Index("ix_chapter_extractions_user_id", "user_id"),
    ]
    if kind:
        columns.append(
            sa.Column("kind", sa.String(length=32), server_default="age", nullable=False)
        )
    return sa.Table("chapter_extractions", sa.MetaData(), *columns)


def upgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table(
            "chapter_extractions", recreate="always", copy_from=_table(kind=False)
        ) as batch:
            batch.add_column(
                sa.Column("kind", sa.String(length=32), server_default="age", nullable=False)
            )
            batch.create_unique_constraint("uq_chapter_extractions_kind", ["chapter_id", "kind"])
    else:
        op.drop_constraint("chapter_extractions_chapter_id_key", "chapter_extractions")
        op.add_column(
            "chapter_extractions",
            sa.Column("kind", sa.String(length=32), server_default="age", nullable=False),
        )
        op.create_unique_constraint(
            "uq_chapter_extractions_kind", "chapter_extractions", ["chapter_id", "kind"]
        )
    op.create_index(
        op.f("ix_chapter_extractions_chapter_id"), "chapter_extractions", ["chapter_id"]
    )


def downgrade() -> None:
    # Only age extractions fit the old table: the others are dropped.
    op.execute("DELETE FROM chapter_extractions WHERE kind != 'age'")
    op.drop_index(op.f("ix_chapter_extractions_chapter_id"), table_name="chapter_extractions")
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table(
            "chapter_extractions", recreate="always", copy_from=_table(kind=True)
        ) as batch:
            batch.drop_column("kind")
            batch.create_unique_constraint("uq_chapter_id", ["chapter_id"])
    else:
        op.drop_constraint("uq_chapter_extractions_kind", "chapter_extractions")
        op.drop_column("chapter_extractions", "kind")
        op.create_unique_constraint(
            "chapter_extractions_chapter_id_key", "chapter_extractions", ["chapter_id"]
        )
