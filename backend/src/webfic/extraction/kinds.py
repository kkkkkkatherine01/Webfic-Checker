"""Kinds of extraction. Each kind has its own prompt, its own model calls and its own
stored reading per chapter (`chapter_extractions`, one row per chapter and kind), and
turns what it read into rows of the fact tables.

Ages are the first kind; step 5-1 adds character facts as a second one instead of
growing the age prompt, so every age baseline stays valid. Adding a kind means adding it
to `KINDS` (docs/steps/05-more-checkers.md A7).
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from webfic.config import Settings
from webfic.db.models import ElapsedTimeFactRow, FactRow
from webfic.extraction import character_facts, extractor
from webfic.extraction.extractor import ChapterExtraction
from webfic.extraction.schemas import RevealedName
from webfic.facts.registry import AGE, APPEARANCE, KINSHIP, LIFE
from webfic.llm.base import LLMClient

# mention, resolved name, quote -> the character's id, or None if it cannot be attributed
Resolve = Callable[[str, str | None, str], uuid.UUID | None]
NameOf = Callable[[uuid.UUID], str]  # a character's canonical name


@dataclass(frozen=True)
class Where:
    """The chapter the rows belong to."""

    user_id: uuid.UUID
    book_id: uuid.UUID
    chapter_id: uuid.UUID
    chapter_number: int


@dataclass
class Rows:
    facts: list[FactRow] = field(default_factory=list)
    spans: list[ElapsedTimeFactRow] = field(default_factory=list)
    unattributed: int = 0  # statements dropped because no character could be named


class ExtractionKind(Protocol):
    name: str  # chapter_extractions.kind

    def version(self, settings: Settings) -> str:
        """Everything besides the text that shapes a reading; a stored reading is
        reused only while it is unchanged."""
        ...

    async def extract(
        self,
        llm: LLMClient,
        *,
        chapter_number: int,
        text: str,
        known_characters: str,
        settings: Settings,
        offset: int,
    ) -> ChapterExtraction: ...

    def dump(self, reading: ChapterExtraction, offset: int) -> dict[str, Any]: ...

    def load(self, data: dict[str, Any], offset: int) -> ChapterExtraction: ...

    def positions_hold(self, reading: ChapterExtraction, content: str) -> bool: ...

    def reveals(self, reading: ChapterExtraction) -> list[RevealedName]:
        """Real names revealed in the chapter, applied before any statement is
        attributed."""
        ...

    def rows(self, reading: Any, *, resolve: Resolve, name_of: NameOf, where: Where) -> Rows: ...


class AgeKind:
    """Ages, time spans and revealed names (prompt `age_v4`)."""

    name = "age"

    def version(self, settings: Settings) -> str:
        return extractor.extraction_version(
            extractor.load_prompt(), settings.chunk_size, settings.chunk_overlap
        )

    async def extract(
        self,
        llm: LLMClient,
        *,
        chapter_number: int,
        text: str,
        known_characters: str,
        settings: Settings,
        offset: int,
    ) -> ChapterExtraction:
        return await extractor.extract_chapter(
            llm, chapter_number=chapter_number, text=text, known_characters=known_characters,
            system_prompt=extractor.load_prompt(), chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap, offset=offset,
        )  # fmt: skip

    def dump(self, reading: ChapterExtraction, offset: int) -> dict[str, Any]:
        return extractor.dump_extraction(reading, offset=offset)

    def load(self, data: dict[str, Any], offset: int) -> ChapterExtraction:
        return extractor.load_extraction(data, offset=offset)

    def positions_hold(self, reading: ChapterExtraction, content: str) -> bool:
        return extractor.positions_hold(reading, content)

    def reveals(self, reading: ChapterExtraction) -> list[RevealedName]:
        return list(reading.revealed_names)

    def rows(
        self, reading: ChapterExtraction, *, resolve: Resolve, name_of: NameOf, where: Where
    ) -> Rows:
        out = Rows()
        for located in reading.ages:
            s = located.statement
            character_id = resolve(s.mention, s.resolved_name, s.raw_text)
            if character_id is None:
                out.unattributed += 1
                continue
            out.facts.append(
                FactRow(
                    user_id=where.user_id, book_id=where.book_id, chapter_id=where.chapter_id,
                    chapter_number=where.chapter_number, character_id=character_id,
                    category=AGE.name, attribute=s.statement_type,
                    mention=s.mention, raw_text=s.raw_text,
                    value_num=s.value, value_max=s.value_max,
                    value_text=s.life_stage.value if s.life_stage else None,
                    is_flashback=s.is_flashback,
                    years_before_present=s.years_before_present,
                    years_before_present_quote=s.years_before_present_quote,
                    is_speculative=s.speculative, qualifiers={},
                    char_start=located.char_start, char_end=located.char_end,
                )
            )  # fmt: skip
        out.spans = [
            ElapsedTimeFactRow(
                user_id=where.user_id, book_id=where.book_id, chapter_id=where.chapter_id,
                chapter_number=where.chapter_number, raw_text=located.statement.raw_text,
                estimated_years=located.statement.estimated_years,
                kind=located.statement.kind, is_flashback=located.statement.is_flashback,
                char_start=located.char_start, char_end=located.char_end,
            )
            for located in reading.elapsed
        ]  # fmt: skip
        return out


class CharacterFactsKind:
    """Fixed features of appearance, blood relations, deaths and appearing in person
    (prompt `character_facts_v2`, steps 5-1 to 5-1e)."""

    name = "character_facts"

    def version(self, settings: Settings) -> str:
        return character_facts.version(
            character_facts.load_prompt(), settings.chunk_size, settings.chunk_overlap
        )

    async def extract(
        self,
        llm: LLMClient,
        *,
        chapter_number: int,
        text: str,
        known_characters: str,
        settings: Settings,
        offset: int,
    ) -> character_facts.FactsReading:
        return await character_facts.extract(
            llm, chapter_number=chapter_number, text=text, known_characters=known_characters,
            system_prompt=character_facts.load_prompt(), chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap, offset=offset,
        )  # fmt: skip

    def dump(self, reading: character_facts.FactsReading, offset: int) -> dict[str, Any]:
        return character_facts.dump(reading, offset=offset)

    def load(self, data: dict[str, Any], offset: int) -> character_facts.FactsReading:
        return character_facts.load(data, offset=offset)

    def positions_hold(self, reading: character_facts.FactsReading, content: str) -> bool:
        return character_facts.positions_hold(reading, content)

    def reveals(self, reading: character_facts.FactsReading) -> list[RevealedName]:
        return []  # real names are revealed by age extraction

    def rows(
        self,
        reading: character_facts.FactsReading,
        *,
        resolve: Resolve,
        name_of: NameOf,
        where: Where,
    ) -> Rows:
        out = Rows()
        for f in reading.facts:
            s = f.statement
            character_id = resolve(s.mention, s.resolved_name, s.raw_text)
            if character_id is None:
                out.unattributed += 1
                continue
            fields: dict[str, Any] = {}
            match f.section:
                case "traits":
                    fields = dict(
                        category=APPEARANCE.name, attribute=s.attribute, value_text=s.value,
                        is_flashback=s.is_flashback, is_speculative=s.speculative,
                        qualifiers={
                            **({"disguised": True} if s.disguised else {}),
                            **({"temporary": True} if s.temporary else {}),
                        },
                    )  # fmt: skip
                case "kinship":
                    other = resolve(s.other_mention, s.other_resolved_name, s.raw_text)
                    if other is None or other == character_id:
                        out.unattributed += 1
                        continue
                    fields = dict(
                        category=KINSHIP.name, attribute=s.relation,
                        is_speculative=s.speculative,
                        qualifiers={
                            "other_name": name_of(other), "other_mention": s.other_mention,
                            **({"address": True} if s.address else {}),
                        },
                    )  # fmt: skip
                case "deaths":
                    fields = dict(
                        category=LIFE.name, attribute="died", is_speculative=s.speculative
                    )
                case "presence":
                    fields = dict(category=LIFE.name, attribute="present")
            out.facts.append(
                FactRow(
                    user_id=where.user_id, book_id=where.book_id, chapter_id=where.chapter_id,
                    chapter_number=where.chapter_number, character_id=character_id,
                    mention=s.mention, raw_text=s.raw_text,
                    char_start=f.char_start, char_end=f.char_end,
                    **{"is_flashback": False, "is_speculative": False, "qualifiers": {}, **fields},
                )
            )  # fmt: skip
        return out


# Every kind, in the order a chapter is read. Ages first: its revealed names are applied
# before any statement is attributed.
KINDS: list[ExtractionKind] = [AgeKind(), CharacterFactsKind()]
