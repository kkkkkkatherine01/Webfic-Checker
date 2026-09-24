from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

SessionFactory = async_sessionmaker


def make_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, pool_pre_ping=True)


def make_session_factory(engine: AsyncEngine) -> SessionFactory:
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def rolled_back(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory whose work is all undone at the end: every session joins one
    outer transaction, so its commits only release savepoints, and the outer transaction
    is rolled back. Used for dry runs. Writes made through other factories (the LLM usage
    ledger) are not part of it and persist."""
    engine: AsyncEngine = factory.kw["bind"]
    async with engine.connect() as connection:
        # The sqlite3 driver manages transactions itself and a SAVEPOINT outside its
        # BEGIN commits on release; for this one connection, let us emit BEGIN (the
        # recipe from the SQLAlchemy SQLite docs). Tests and evaluation runs use SQLite.
        sqlite = engine.dialect.name == "sqlite"
        if sqlite:
            saved = await connection.run_sync(_set_isolation_level, None)
        outer = await connection.begin()
        if sqlite:
            await connection.exec_driver_sql("BEGIN")
        try:
            yield async_sessionmaker(
                bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
            )
        finally:
            await outer.rollback()
            if sqlite:
                await connection.run_sync(_set_isolation_level, saved)


def _set_isolation_level(sync_connection: Connection, level: str | None) -> str | None:
    """Set the sqlite3 driver's isolation level on this connection; returns the old one."""
    dbapi = sync_connection.connection.dbapi_connection
    assert dbapi is not None
    saved = dbapi.isolation_level
    dbapi.isolation_level = level
    return saved
