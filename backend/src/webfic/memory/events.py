"""Change log of the character table, and undoing it.

Extraction creates characters, adds aliases, renames (a real name is revealed) and
merges (a character turns out to be someone already known). Each change is logged with
its chapter. Recomputing from chapter N undoes the changes of chapters N.. newest first,
which leaves the character table exactly as it was after chapter N-1.

Undo does not rely on foreign-key cascades (SQLite, used in tests, does not enforce
them by default): dependent rows are removed explicitly.
"""

import uuid
from enum import StrEnum
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from webfic.db.models import (
    Character,
    CharacterAlias,
    CharacterEvent,
    CharacterStateRow,
    FactRow,
)


class EventKind(StrEnum):
    CREATE = "create"  # {character_id, name}
    ALIAS = "alias"  # {alias_id, character_id, alias}
    RENAME = "rename"  # {character_id, old_name, new_name}
    MERGE = "merge"  # {from_id, from_name, into_id, fact_ids, alias_ids}


async def record(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    chapter_id: uuid.UUID,
    chapter_number: int,
    events: list[tuple[EventKind, dict[str, Any]]],
) -> None:
    """Log one chapter's changes, in the order they were applied."""
    if not events:
        return
    last = await session.scalar(
        select(func.max(CharacterEvent.seq)).where(CharacterEvent.book_id == book_id)
    )
    for offset, (kind, payload) in enumerate(events, start=1):
        session.add(
            CharacterEvent(
                user_id=user_id, book_id=book_id, seq=(last or 0) + offset,
                chapter_id=chapter_id, chapter_number=chapter_number,
                source="extracted", kind=kind, payload=payload,
            )
        )  # fmt: skip


async def undo_from_chapter(
    session: AsyncSession, *, user_id: uuid.UUID, book_id: uuid.UUID, chapter_number: int
) -> int:
    """Undo, newest first, every extracted change made in chapter `chapter_number` or
    later, and drop those log entries. Returns how many changes were undone."""
    events = (
        await session.scalars(
            select(CharacterEvent)
            .where(
                CharacterEvent.user_id == user_id,
                CharacterEvent.book_id == book_id,
                CharacterEvent.source == "extracted",
                CharacterEvent.chapter_number >= chapter_number,
            )
            .order_by(CharacterEvent.seq.desc())
        )
    ).all()
    for event in events:
        await _undo(session, user_id, book_id, EventKind(event.kind), event.payload)
        await session.delete(event)
    await session.flush()
    return len(events)


async def _undo(
    session: AsyncSession,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    kind: EventKind,
    payload: dict[str, Any],
) -> None:
    def ids(key: str) -> list[uuid.UUID]:
        return [uuid.UUID(i) for i in payload.get(key, [])]

    match kind:
        case EventKind.CREATE:
            character_id = uuid.UUID(payload["character_id"])
            for model in (FactRow, CharacterAlias, CharacterStateRow):
                await session.execute(delete(model).where(model.character_id == character_id))
            await session.execute(
                delete(Character).where(
                    Character.id == character_id,
                    Character.user_id == user_id,
                    Character.book_id == book_id,
                )
            )
        case EventKind.ALIAS:
            await session.execute(
                delete(CharacterAlias).where(
                    CharacterAlias.id == uuid.UUID(payload["alias_id"]),
                    CharacterAlias.user_id == user_id,
                )
            )
        case EventKind.RENAME:
            await session.execute(
                update(Character)
                .where(
                    Character.id == uuid.UUID(payload["character_id"]),
                    Character.user_id == user_id,
                )
                .values(canonical_name=payload["old_name"])
            )
        case EventKind.MERGE:
            from_id = uuid.UUID(payload["from_id"])
            session.add(
                Character(
                    id=from_id, user_id=user_id, book_id=book_id,
                    canonical_name=payload["from_name"],
                )
            )  # fmt: skip
            await session.flush()
            # Rows of later chapters may be gone already; the rest move back.
            for model, key in ((FactRow, "fact_ids"), (CharacterAlias, "alias_ids")):
                if moved := ids(key):
                    await session.execute(
                        update(model)
                        .where(model.id.in_(moved), model.user_id == user_id)
                        .values(character_id=from_id)
                    )
