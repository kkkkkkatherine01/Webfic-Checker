"""Adapter for the OpenAI-compatible chat protocol: OpenAI, DeepSeek, Qwen, Kimi, ..."""

import re
from typing import Any

import openai
from openai import AsyncOpenAI

from webfic.llm.base import (
    ChatMessage,
    ProviderError,
    ProviderErrorKind,
    RawCompletion,
    Usage,
)

# Providers signal moderation refusals differently; DeepSeek, for instance, answers 400
# with "Content Exists Risk".
_CONTENT_FILTER_HINTS = re.compile(
    r"content.?(exists.?)?risk|content.?filter|moderation|sensitive|inappropriate|审核|敏感",
    re.IGNORECASE,
)


class OpenAICompatBackend:
    def __init__(self, *, base_url: str, api_key: str, provider: str | None = None):
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.provider = provider or base_url

    async def chat(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        json_mode: bool,
        extra: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> RawCompletion:
        optional: dict[str, Any] = {}
        if temperature is not None:
            optional["temperature"] = temperature
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in messages],  # type: ignore[misc]
                response_format={"type": "json_object"} if json_mode else {"type": "text"},
                extra_body=extra or None,
                **optional,
            )
        except openai.APIError as exc:
            raise translate_error(exc) from exc

        choice = response.choices[0]
        if choice.finish_reason == "content_filter":
            raise ProviderError(ProviderErrorKind.CONTENT_FILTER, "输出被服务商的内容审核拦截")
        text = choice.message.content or ""
        return RawCompletion(text=text, usage=_usage(response.usage), model=response.model)


def translate_error(exc: openai.APIError) -> ProviderError:
    """Map OpenAI SDK exceptions (after the SDK's own retries) to ProviderError."""
    message = str(getattr(exc, "message", None) or exc)
    if _CONTENT_FILTER_HINTS.search(message):
        kind = ProviderErrorKind.CONTENT_FILTER
    elif isinstance(exc, openai.RateLimitError):
        kind = ProviderErrorKind.RATE_LIMIT
    elif isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        kind = ProviderErrorKind.AUTH
    elif isinstance(exc, openai.APIStatusError) and exc.status_code == 402:
        kind = ProviderErrorKind.AUTH  # DeepSeek: insufficient balance
    elif isinstance(exc, openai.APIConnectionError):  # includes timeouts
        kind = ProviderErrorKind.NETWORK
    elif isinstance(exc, openai.InternalServerError):
        kind = ProviderErrorKind.SERVER
    else:
        kind = ProviderErrorKind.BAD_REQUEST
    return ProviderError(kind, message[:500])


def _usage(usage: Any) -> Usage:
    if usage is None:
        return Usage()
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None and getattr(details, "cached_tokens", None):
        cached = details.cached_tokens  # OpenAI style
    extra = getattr(usage, "model_extra", None) or {}
    if not cached and extra.get("prompt_cache_hit_tokens"):
        cached = int(extra["prompt_cache_hit_tokens"])  # DeepSeek style
    return Usage(
        input_tokens=usage.prompt_tokens or 0,
        cached_input_tokens=cached,
        output_tokens=usage.completion_tokens or 0,
    )
