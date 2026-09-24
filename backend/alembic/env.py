import asyncio

from alembic import context
from sqlalchemy.engine import Connection

from webfic.config import get_settings
from webfic.db.models import Base
from webfic.db.session import make_engine

target_metadata = Base.metadata

# Postgres-only expression indexes, created by hand in migrations and absent from the
# models (SQLite, used in tests, cannot build them); autogenerate must leave them alone.
_MIGRATION_ONLY_INDEXES = {"ix_passages_tokens"}


def _include_object(obj, name, type_, reflected, compare_to) -> bool:
    return not (type_ == "index" and name in _MIGRATION_ONLY_INDEXES)


def _run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection, target_metadata=target_metadata, include_object=_include_object
    )
    with context.begin_transaction():
        context.run_migrations()


async def _run_online() -> None:
    engine = make_engine(get_settings().database_url)
    async with engine.connect() as connection:
        await connection.run_sync(_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    context.configure(url=get_settings().database_url, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(_run_online())
