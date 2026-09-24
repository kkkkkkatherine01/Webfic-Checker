"""generic facts table (age_facts moved in)

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-24 21:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")

# Columns shared by both tables, copied as they are (ids included).
_COMMON = (
    "id, created_at, user_id, book_id, chapter_id, chapter_number, raw_text, is_flashback,"
    " char_start, char_end, character_id, mention, value_max, years_before_present,"
    " years_before_present_quote, is_speculative"
)


def upgrade() -> None:
    op.create_table(
        "facts",
        sa.Column("character_id", sa.Uuid(), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("attribute", sa.String(length=32), nullable=False),
        sa.Column("mention", sa.Text(), nullable=False),
        sa.Column("value_num", sa.Float(), nullable=True),
        sa.Column("value_max", sa.Float(), nullable=True),
        sa.Column("value_text", sa.Text(), nullable=True),
        sa.Column("years_before_present", sa.Float(), nullable=True),
        sa.Column("years_before_present_quote", sa.Text(), nullable=True),
        sa.Column("is_speculative", sa.Boolean(), nullable=False),
        # The default is only for the rows moved in below; new rows always set it.
        sa.Column("qualifiers", _JSON, nullable=False, server_default=sa.text("'{}'")),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("book_id", sa.Uuid(), nullable=False),
        sa.Column("chapter_id", sa.Uuid(), nullable=False),
        sa.Column("chapter_number", sa.Integer(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("is_flashback", sa.Boolean(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["book_id"], ["books.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chapter_id"], ["chapters.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["character_id"], ["characters.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_facts_book_character_category",
        "facts",
        ["book_id", "character_id", "category"],
        unique=False,
    )
    op.create_index(op.f("ix_facts_book_id"), "facts", ["book_id"], unique=False)
    op.create_index(op.f("ix_facts_chapter_id"), "facts", ["chapter_id"], unique=False)
    op.create_index(op.f("ix_facts_user_id"), "facts", ["user_id"], unique=False)

    # Every existing fact is an age statement: statement_type becomes the attribute, the
    # value moves to value_num and the life stage to value_text.
    op.execute(
        f"INSERT INTO facts ({_COMMON}, category, attribute, value_num, value_text) "
        f"SELECT {_COMMON}, 'age', statement_type, value, life_stage FROM age_facts"
    )

    op.drop_index(op.f("ix_age_facts_chapter_id"), table_name="age_facts")
    op.drop_index(op.f("ix_age_facts_user_id"), table_name="age_facts")
    op.drop_index(op.f("ix_age_facts_book_id"), table_name="age_facts")
    op.drop_table("age_facts")


def downgrade() -> None:
    op.create_table(
        "age_facts",
        sa.Column("character_id", sa.Uuid(), nullable=False),
        sa.Column("mention", sa.Text(), nullable=False),
        sa.Column("statement_type", sa.String(length=16), nullable=False),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("life_stage", sa.String(length=16), nullable=True),
        sa.Column("years_before_present", sa.Float(), nullable=True),
        sa.Column("value_max", sa.Float(), nullable=True),
        sa.Column("is_speculative", sa.Boolean(), nullable=False),
        sa.Column("years_before_present_quote", sa.Text(), nullable=True),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("book_id", sa.Uuid(), nullable=False),
        sa.Column("chapter_id", sa.Uuid(), nullable=False),
        sa.Column("chapter_number", sa.Integer(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("is_flashback", sa.Boolean(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["book_id"], ["books.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chapter_id"], ["chapters.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["character_id"], ["characters.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_age_facts_book_id"), "age_facts", ["book_id"], unique=False)
    op.create_index(op.f("ix_age_facts_user_id"), "age_facts", ["user_id"], unique=False)
    op.create_index(op.f("ix_age_facts_chapter_id"), "age_facts", ["chapter_id"], unique=False)

    # Only age facts fit the old table; other categories cannot exist before this revision.
    op.execute(
        f"INSERT INTO age_facts ({_COMMON}, statement_type, value, life_stage) "
        f"SELECT {_COMMON}, attribute, value_num, value_text FROM facts WHERE category = 'age'"
    )

    op.drop_index(op.f("ix_facts_user_id"), table_name="facts")
    op.drop_index(op.f("ix_facts_chapter_id"), table_name="facts")
    op.drop_index(op.f("ix_facts_book_id"), table_name="facts")
    op.drop_index("ix_facts_book_character_category", table_name="facts")
    op.drop_table("facts")
