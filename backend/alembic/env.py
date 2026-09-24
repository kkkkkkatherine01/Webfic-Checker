import asyncio

from alembic import context
from sqlalchemy.engine import Connection

from webfic.config import get_settings
from webfic.db.models import Base
from webfic.db.session import make_engine

target_metadata = Base.metadata


def _run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
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
