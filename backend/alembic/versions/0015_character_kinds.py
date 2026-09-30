"""character kinds: which kind of extraction named a character or alias

Revision ID: 0015
Revises: 0014
Create Date: 2026-09-27

Step 5-1: each kind of extraction is shown only the characters it named, so adding a kind
never changes what the others are asked. Every existing character and alias was named by
age extraction.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for table in ("characters", "character_aliases"):
        op.add_column(
            table, sa.Column("kind", sa.String(length=32), server_default="age", nullable=False)
        )


def downgrade() -> None:
    for table in ("characters", "character_aliases"):
        with op.batch_alter_table(table) as batch:
            batch.drop_column("kind")
