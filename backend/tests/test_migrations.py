"""Run the Alembic migrations on a throw-away SQLite database: the whole chain applies,
and 0006 moves age facts into the generic facts table and back without losing data."""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]

BOOK, CHAPTER, CHARACTER, FACT = "b" * 32, "c" * 32, "d" * 32, "e" * 32
USER = "a" * 32


def alembic(db: Path, *args: str) -> None:
    env = {**os.environ, "WEBFIC_DATABASE_URL": f"sqlite+aiosqlite:///{db.as_posix()}"}
    subprocess.run(
        [sys.executable, "-m", "alembic", *args], cwd=BACKEND, env=env, check=True,
        capture_output=True,
    )  # fmt: skip


def rows(db: Path, query: str) -> list[tuple]:
    with sqlite3.connect(db) as conn:
        return conn.execute(query).fetchall()


def test_0006_moves_age_facts_into_facts_and_back(tmp_path):
    db = tmp_path / "m.db"
    alembic(db, "upgrade", "0005")
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO books (id, user_id, title) VALUES (?, ?, '测试')", (BOOK, USER))
        conn.execute(
            "INSERT INTO chapters (id, user_id, book_id, number, title, content, char_count,"
            " content_hash, status) VALUES (?, ?, ?, 1, '第一章', '正文', 2, 'h', 'extracted')",
            (CHAPTER, USER, BOOK),
        )
        conn.execute(
            "INSERT INTO characters (id, user_id, book_id, canonical_name)"
            " VALUES (?, ?, ?, '林远')",
            (CHARACTER, USER, BOOK),
        )
        conn.execute(
            "INSERT INTO age_facts (id, user_id, book_id, chapter_id, chapter_number,"
            " character_id, mention, raw_text, statement_type, value, value_max, life_stage,"
            " is_flashback, years_before_present, years_before_present_quote, is_speculative,"
            " char_start, char_end) VALUES (?, ?, ?, ?, 1, ?, '林远', '林远三十来岁',"
            " 'absolute_age', 30, 39, NULL, 1, 15, '十五年前', 0, 0, 6)",
            (FACT, USER, BOOK, CHAPTER, CHARACTER),
        )

    alembic(db, "upgrade", "0006")
    assert rows(db, "SELECT name FROM sqlite_master WHERE name = 'age_facts'") == []
    assert rows(
        db,
        "SELECT id, category, attribute, value_num, value_max, value_text, is_flashback,"
        " years_before_present, years_before_present_quote, qualifiers FROM facts",
    ) == [(FACT, "age", "absolute_age", 30, 39, None, 1, 15, "十五年前", "{}")]

    alembic(db, "downgrade", "0005")
    assert rows(db, "SELECT name FROM sqlite_master WHERE name = 'facts'") == []
    assert rows(
        db,
        "SELECT id, statement_type, value, value_max, life_stage, is_flashback,"
        " years_before_present, years_before_present_quote FROM age_facts",
    ) == [(FACT, "absolute_age", 30, 39, None, 1, 15, "十五年前")]

    alembic(db, "upgrade", "head")
    assert rows(db, "SELECT id, value_num FROM facts") == [(FACT, 30)]


def test_tables_added_after_0006_come_and_go(tmp_path):
    db = tmp_path / "m.db"
    tables = "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN"
    names = "('character_events', 'character_states', 'core_snapshots', 'chapter_extractions')"
    alembic(db, "upgrade", "head")
    assert len(rows(db, f"{tables} {names}")) == 4
    alembic(db, "downgrade", "0006")
    assert rows(db, f"{tables} {names}") == []
    alembic(db, "upgrade", "head")
    assert len(rows(db, f"{tables} {names}")) == 4
