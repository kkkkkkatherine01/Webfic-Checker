"""character change log and core state

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-24 22:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def _common() -> list[sa.Column]:
    return [
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("book_id", sa.Uuid(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "character_events",
        *_common(),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("chapter_id", sa.Uuid(), nullable=True),
        sa.Column("chapter_number", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("payload", _JSON, nullable=False),
        sa.ForeignKeyConstraint(["book_id"], ["books.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chapter_id"], ["chapters.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("book_id", "seq"),
    )
    op.create_index(
        op.f("ix_character_events_chapter_id"), "character_events", ["chapter_id"], unique=False
    )
    op.create_index(
        op.f("ix_character_events_user_id"), "character_events", ["user_id"], unique=False
    )

    op.create_table(
        "character_states",
        *_common(),
        sa.Column("character_id", sa.Uuid(), nullable=False),
        sa.Column("chapter_number", sa.Integer(), nullable=False),
        sa.Column("state", _JSON, nullable=False),
        sa.ForeignKeyConstraint(["book_id"], ["books.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["character_id"], ["characters.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("character_id"),
    )
    op.create_index(
        op.f("ix_character_states_book_id"), "character_states", ["book_id"], unique=False
    )
    op.create_index(
        op.f("ix_character_states_user_id"), "character_states", ["user_id"], unique=False
    )

    op.create_table(
        "core_snapshots",
        *_common(),
        sa.Column("chapter_id", sa.Uuid(), nullable=False),
        sa.Column("chapter_number", sa.Integer(), nullable=False),
        sa.Column("state", _JSON, nullable=False),
        sa.ForeignKeyConstraint(["book_id"], ["books.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chapter_id"], ["chapters.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chapter_id"),
    )
    op.create_index(op.f("ix_core_snapshots_book_id"), "core_snapshots", ["book_id"], unique=False)
    op.create_index(op.f("ix_core_snapshots_user_id"), "core_snapshots", ["user_id"], unique=False)
    # Books extracted before this revision have no log and no Core state: re-import them,
    # or rebuild the Core from their facts (webfic.memory.core.rebuild).


def downgrade() -> None:
    op.drop_index(op.f("ix_core_snapshots_user_id"), table_name="core_snapshots")
    op.drop_index(op.f("ix_core_snapshots_book_id"), table_name="core_snapshots")
    op.drop_table("core_snapshots")
    op.drop_index(op.f("ix_character_states_user_id"), table_name="character_states")
    op.drop_index(op.f("ix_character_states_book_id"), table_name="character_states")
    op.drop_table("character_states")
    op.drop_index(op.f("ix_character_events_user_id"), table_name="character_events")
    op.drop_index(op.f("ix_character_events_chapter_id"), table_name="character_events")
    op.drop_table("character_events")
