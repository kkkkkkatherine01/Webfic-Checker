"""The fact category registry and the Recall queries over the facts table."""

import uuid

import pytest

from tests.fakes import FakeBackend
from tests.test_pipeline import USER, import_book, respond
from webfic.facts.registry import (
    AGE,
    ChangeRule,
    UnknownFactKind,
    change_rule,
    validate_qualifiers,
)
from webfic.memory import recall
from webfic.services.errors import NotFound

# --- registry --------------------------------------------------------------------------


def test_age_attributes_follow_story_time():
    for attribute in ("absolute_age", "relative_age", "birth_year", "life_stage"):
        assert change_rule(AGE.name, attribute) is ChangeRule.TEMPORAL


def test_unregistered_kinds_are_rejected():
    with pytest.raises(UnknownFactKind):
        change_rule("appearance", "eye_color")  # step 5
    with pytest.raises(UnknownFactKind):
        change_rule(AGE.name, "height")
    with pytest.raises(UnknownFactKind):
        validate_qualifiers("appearance", {})


def test_qualifiers_are_validated_by_the_category_model():
    assert validate_qualifiers(AGE.name, {}) == {}
    with pytest.raises(ValueError):
        validate_qualifiers(AGE.name, {"unexpected": True})


# --- recall queries --------------------------------------------------------------------


@pytest.fixture
async def book(factory):
    job, _, _ = await import_book(factory, FakeBackend(respond))
    return job.book_id


async def test_facts_are_stored_as_age_category(factory, book):
    async with factory() as session:
        facts = await recall.list_facts(session, user_id=USER, book_id=book)
    assert [(f.chapter_number, f.category, f.attribute, f.value_num) for f in facts] == [
        (1, "age", "absolute_age", 18),
        (2, "age", "absolute_age", 21),
        (3, "age", "absolute_age", 16),
    ]
    assert facts[1].raw_text == "林少爷二十一岁了" and facts[1].mention == "林少爷"
    assert all(f.character_name == "林远" and f.qualifiers == {} for f in facts)


async def test_list_facts_filters(factory, book):
    async def chapters_of(**filters):
        async with factory() as session:
            facts = await recall.list_facts(session, user_id=USER, book_id=book, **filters)
        return [f.chapter_number for f in facts]

    assert await chapters_of(character="林少爷") == [1, 2, 3]  # an alias finds the character
    assert await chapters_of(category="age", chapters=(2, 3)) == [2, 3]
    assert await chapters_of(category="appearance") == []
    assert await chapters_of(chapters=(4, 9)) == []


async def test_unknown_character_is_reported(factory, book):
    async with factory() as session:
        with pytest.raises(NotFound):
            await recall.list_facts(session, user_id=USER, book_id=book, character="苏晚晴")


async def test_list_time_spans(factory, book):
    async with factory() as session:
        spans = await recall.list_time_spans(session, user_id=USER, book_id=book)
        none = await recall.list_time_spans(session, user_id=USER, book_id=book, chapters=(3, 3))
    assert [(s.chapter_number, s.raw_text, s.estimated_years, s.kind) for s in spans] == [
        (2, "三年后", 3, "advance")
    ]
    assert none == []


async def test_queries_never_cross_users(factory, book):
    stranger = uuid.uuid4()
    async with factory() as session:
        with pytest.raises(NotFound):
            await recall.list_facts(session, user_id=stranger, book_id=book)
        with pytest.raises(NotFound):
            await recall.list_time_spans(session, user_id=stranger, book_id=book)
        with pytest.raises(NotFound):
            await recall.find_character(session, user_id=stranger, book_id=book, name="林远")
