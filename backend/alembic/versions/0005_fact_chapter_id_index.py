"""fact chapter_id index

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-24 20:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Re-extracting a chapter deletes its facts by chapter_id.
    op.create_index(op.f("ix_age_facts_chapter_id"), "age_facts", ["chapter_id"], unique=False)
    op.create_index(
        op.f("ix_elapsed_time_facts_chapter_id"), "elapsed_time_facts", ["chapter_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_elapsed_time_facts_chapter_id"), table_name="elapsed_time_facts")
    op.drop_index(op.f("ix_age_facts_chapter_id"), table_name="age_facts")
