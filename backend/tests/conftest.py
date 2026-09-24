import pytest

from webfic.db.models import Base
from webfic.db.session import make_engine, make_session_factory


@pytest.fixture
async def factory(tmp_path):
    """Session factory on a fresh SQLite database."""
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield make_session_factory(engine)
    await engine.dispose()
