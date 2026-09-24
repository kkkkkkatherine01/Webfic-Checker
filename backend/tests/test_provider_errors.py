import httpx2 as httpx  # the openai SDK (v3) is built on httpx2
import openai
import pytest

from tests import fakes
from tests.fakes import FakeBackend
from tests.test_pipeline import BOOK, USER, respond
from webfic.config import Settings
from webfic.db.models import Chapter
from webfic.llm.base import ProviderError, ProviderErrorKind
from webfic.llm.openai_compat import translate_error
from webfic.services import imports

REQUEST = httpx.Request("POST", "https://api.example.com/chat/completions")


def status_error(cls, status: int, message: str):
    response = httpx.Response(status, request=REQUEST, json={"error": {"message": message}})
    return cls(message, response=response, body=None)


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (status_error(openai.BadRequestError, 400, "Content Exists Risk"), "content_filter"),
        (status_error(openai.BadRequestError, 400, "输入内容包含敏感信息"), "content_filter"),
        (status_error(openai.BadRequestError, 400, "max_tokens too large"), "bad_request"),
        (status_error(openai.RateLimitError, 429, "Rate limit reached"), "rate_limit"),
        (status_error(openai.AuthenticationError, 401, "Invalid API key"), "auth"),
        (status_error(openai.APIStatusError, 402, "Insufficient Balance"), "auth"),
        (status_error(openai.InternalServerError, 503, "Server overloaded"), "server"),
        (openai.APIConnectionError(request=REQUEST), "network"),
        (openai.APITimeoutError(request=REQUEST), "network"),
    ],
)
def test_translate_error(exc, kind):
    assert translate_error(exc).kind == kind


class RefusingBackend(FakeBackend):
    """Refuses any chunk containing `trigger` with the given provider error."""

    def __init__(self, trigger: str, kind: ProviderErrorKind):
        super().__init__(respond)
        self._trigger, self._kind = trigger, kind

    async def chat(self, *, model, messages, json_mode, extra=None, temperature=None):
        if self._trigger in messages[1].content:
            raise ProviderError(self._kind, "refused")
        return await super().chat(model=model, messages=messages, json_mode=json_mode)


async def run(factory, backend):
    async with factory() as session:
        job = await imports.create_import_job(session, user_id=USER, title="t", text=BOOK)
    llm = fakes.make_llm(backend, factory, user_id=USER, book_id=job.book_id)
    return job, await imports.run_import_job(
        factory, llm, Settings(), user_id=USER, book_id=job.book_id
    )


async def test_refused_chapter_fails_alone_and_the_rest_continue(factory):
    _, result = await run(factory, RefusingBackend("第 2 章", ProviderErrorKind.CONTENT_FILTER))
    assert (result.extracted, result.failed) == (2, 1)
    assert result.failures == {"content_filter": 1}

    async with factory() as session:
        chapter = await session.scalar(
            Chapter.__table__.select().where(Chapter.number == 2).with_only_columns(Chapter.error)
        )
    assert chapter.startswith("[content_filter]")


async def test_auth_error_stops_the_whole_import(factory):
    with pytest.raises(ProviderError) as exc:
        await run(factory, RefusingBackend("第 1 章", ProviderErrorKind.AUTH))
    assert exc.value.kind == ProviderErrorKind.AUTH
