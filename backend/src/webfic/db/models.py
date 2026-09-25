"""SQLAlchemy models. Every business table carries `user_id`; book-scoped tables also
carry `book_id`. `user_id` has no foreign key until the users table exists (before launch)."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

JsonType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class _Common:
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Book(_Common, Base):
    __tablename__ = "books"

    user_id: Mapped[uuid.UUID] = mapped_column(index=True)
    title: Mapped[str] = mapped_column(Text)


class Chapter(_Common, Base):
    __tablename__ = "chapters"
    __table_args__ = (UniqueConstraint("book_id", "number"),)

    user_id: Mapped[uuid.UUID] = mapped_column(index=True)
    book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    number: Mapped[int]
    title: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int]
    content_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    error: Mapped[str | None] = mapped_column(Text)


class Character(_Common, Base):
    __tablename__ = "characters"

    user_id: Mapped[uuid.UUID] = mapped_column(index=True)
    book_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("books.id", ondelete="CASCADE"), index=True
    )
    canonical_name: Mapped[str] = mapped_column(Text)


class CharacterAlias(_Common, Base):
    __tablename__ = "character_aliases"
    __table_args__ = (UniqueConstraint("book_id", "alias"),)

    user_id: Mapped[uuid.UUID] = mapped_column(index=True)
    book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    character_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("characters.id", ondelete="CASCADE"))
    alias: Mapped[str] = mapped_column(Text)
    first_chapter: Mapped[int]
    source: Mapped[str] = mapped_column(String(16), default="extracted")


class CharacterEvent(_Common, Base):
    """Change log of the character table: every character created, alias added, rename
    and merge made while extracting a chapter. Recomputing from chapter N undoes the
    events of chapters N.. in reverse `seq` order (see webfic.memory.events)."""

    __tablename__ = "character_events"
    __table_args__ = (UniqueConstraint("book_id", "seq"),)

    user_id: Mapped[uuid.UUID] = mapped_column(index=True)
    book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    seq: Mapped[int]  # order within the book
    chapter_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("chapters.id", ondelete="CASCADE"), index=True
    )
    chapter_number: Mapped[int | None]  # None for the author's own edits (before launch)
    source: Mapped[str] = mapped_column(String(16), default="extracted")  # extracted / user
    kind: Mapped[str] = mapped_column(String(16))  # create / alias / rename / merge
    payload: Mapped[dict[str, Any]] = mapped_column(JsonType)


class CharacterStateRow(_Common, Base):
    """Core layer: the latest known state of each character (a `CharacterState`)."""

    __tablename__ = "character_states"

    user_id: Mapped[uuid.UUID] = mapped_column(index=True)
    book_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("books.id", ondelete="CASCADE"), index=True
    )
    character_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("characters.id", ondelete="CASCADE"), unique=True
    )
    chapter_number: Mapped[int]  # the state is as of the end of this chapter
    state: Mapped[dict[str, Any]] = mapped_column(JsonType)


class CoreSnapshot(_Common, Base):
    """The whole book's Core state (a `BookState`) at the end of one chapter, for rolling
    back when a chapter changes and for "what was known by chapter N" queries."""

    __tablename__ = "core_snapshots"

    user_id: Mapped[uuid.UUID] = mapped_column(index=True)
    book_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("books.id", ondelete="CASCADE"), index=True
    )
    chapter_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chapters.id", ondelete="CASCADE"), unique=True
    )
    chapter_number: Mapped[int]
    state: Mapped[dict[str, Any]] = mapped_column(JsonType)


class ChapterExtractionRow(_Common, Base):
    """What the model extracted from one chapter (all chunks merged, quotes located),
    kept so that recomputing reuses it while the chapter text and the extraction setup
    are unchanged. Asking the model again would give a slightly different reading and
    blur what an edit really changed."""

    __tablename__ = "chapter_extractions"

    user_id: Mapped[uuid.UUID] = mapped_column(index=True)
    book_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("books.id", ondelete="CASCADE"), index=True
    )
    chapter_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chapters.id", ondelete="CASCADE"), unique=True
    )
    content_hash: Mapped[str] = mapped_column(String(64))  # of the chapter text read
    version: Mapped[str] = mapped_column(String(32))  # prompt and chunking used
    result: Mapped[dict[str, Any]] = mapped_column(JsonType)


EMBEDDING_DIM = 512  # BAAI/bge-small-zh-v1.5


class PassageRow(_Common, Base):
    """Archival layer: a short passage of a chapter's text (about 500 characters, cut at
    sentence ends, overlapping its neighbours), with its embedding and its jieba tokens
    for keyword search. Passages depend only on their chapter's text."""

    __tablename__ = "passages"
    __table_args__ = (Index("ix_passages_book_chapter", "book_id", "chapter_number"),)

    user_id: Mapped[uuid.UUID] = mapped_column(index=True)
    book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    chapter_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chapters.id", ondelete="CASCADE"), index=True
    )
    chapter_number: Mapped[int]
    # Chapter text + passage size/overlap + embedding model: passages are rebuilt only
    # when one of these changes.
    source_key: Mapped[str] = mapped_column(String(64))
    char_start: Mapped[int]
    char_end: Mapped[int]
    text: Mapped[str] = mapped_column(Text)
    tokens: Mapped[str] = mapped_column(Text)  # space-separated jieba tokens
    # pgvector on Postgres; plain JSON on SQLite, where unit tests search in Python.
    embedding: Mapped[Any] = mapped_column(JSON().with_variant(Vector(EMBEDDING_DIM), "postgresql"))


class _ChapterFact(_Common):
    user_id: Mapped[uuid.UUID] = mapped_column(index=True)
    book_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("books.id", ondelete="CASCADE"), index=True
    )
    # Indexed: re-extracting a chapter deletes its facts by chapter.
    chapter_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chapters.id", ondelete="CASCADE"), index=True
    )
    chapter_number: Mapped[int]
    raw_text: Mapped[str] = mapped_column(Text)
    is_flashback: Mapped[bool] = mapped_column(default=False)
    char_start: Mapped[int]
    char_end: Mapped[int]


class FactRow(_ChapterFact, Base):
    """One statement about a character: an age today, appearance, a title... Categories
    and their attributes are registered in `webfic.facts.registry`."""

    __tablename__ = "facts"
    __table_args__ = (
        Index("ix_facts_book_character_category", "book_id", "character_id", "category"),
    )

    character_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("characters.id", ondelete="CASCADE"))
    category: Mapped[str] = mapped_column(String(32))
    # For ages the statement type: absolute_age / relative_age / birth_year / life_stage.
    attribute: Mapped[str] = mapped_column(String(32))
    mention: Mapped[str] = mapped_column(Text)
    value_num: Mapped[float | None]
    value_max: Mapped[float | None]  # upper end of an approximate value ("三十来岁")
    value_text: Mapped[str | None] = mapped_column(Text)  # e.g. the life stage
    # When and how reliable, whatever the category: a statement may sit in a flashback
    # (how long ago, and the text saying so) or be a guess, hypothetical or hearsay.
    years_before_present: Mapped[float | None]
    years_before_present_quote: Mapped[str | None] = mapped_column(Text)
    is_speculative: Mapped[bool] = mapped_column(default=False)
    # Category-specific extras, validated by the category's model in the registry.
    qualifiers: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)


class ElapsedTimeFactRow(_ChapterFact, Base):
    __tablename__ = "elapsed_time_facts"

    estimated_years: Mapped[float | None]
    kind: Mapped[str] = mapped_column(String(16), default="advance")


class IssueRow(_Common, Base):
    __tablename__ = "consistency_issues"
    __table_args__ = (UniqueConstraint("book_id", "fingerprint"),)

    user_id: Mapped[uuid.UUID] = mapped_column(index=True)
    book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    checker: Mapped[str] = mapped_column(String(32))
    issue_type: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[str] = mapped_column(String(24))
    status: Mapped[str] = mapped_column(String(16), default="open")
    description: Mapped[str] = mapped_column(Text)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JsonType)
    subjects: Mapped[list[str]] = mapped_column(JsonType, default=list)  # character ids
    fingerprint: Mapped[str] = mapped_column(String(32))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class LLMCallRow(_Common, Base):
    """One row per LLM request: the usage ledger and the response cache."""

    __tablename__ = "llm_calls"
    __table_args__ = (Index("ix_llm_calls_request_hash_ok", "request_hash", "ok"),)

    user_id: Mapped[uuid.UUID | None] = mapped_column(index=True)
    book_id: Mapped[uuid.UUID | None] = mapped_column(index=True)
    purpose: Mapped[str] = mapped_column(String(64))
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(64))
    request_hash: Mapped[str] = mapped_column(String(64))
    response_text: Mapped[str] = mapped_column(Text)
    input_tokens: Mapped[int] = mapped_column(default=0)
    cached_input_tokens: Mapped[int] = mapped_column(default=0)
    output_tokens: Mapped[int] = mapped_column(default=0)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), default=Decimal(0))
    latency_ms: Mapped[int] = mapped_column(default=0)
    cache_hit: Mapped[bool] = mapped_column(default=False)
    ok: Mapped[bool] = mapped_column(default=False)
