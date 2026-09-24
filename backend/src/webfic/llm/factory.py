"""Build the LLMClient from platform settings. User BYOK configs are added here
before launch."""

from webfic.config import Settings
from webfic.llm.base import LLMClient, Tier
from webfic.llm.client import CallStore, JsonLLMClient, TierConfig
from webfic.llm.openai_compat import OpenAICompatBackend


class LLMNotConfigured(Exception):
    pass


def platform_client(settings: Settings, store: CallStore | None = None) -> LLMClient:
    if settings.llm_api_key is None or not settings.llm_api_key.get_secret_value().strip():
        raise LLMNotConfigured(
            "未配置 LLM API Key：请在项目根目录的 .env 中设置 WEBFIC_LLM_API_KEY"
        )
    backend = OpenAICompatBackend(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
    )
    tiers = {
        Tier.EXTRACT: TierConfig(
            settings.llm_extract_model,
            settings.llm_extract_extra,
            settings.llm_extract_temperature,
        ),
        Tier.REASON: TierConfig(
            settings.llm_reason_model,
            settings.llm_reason_extra,
            settings.llm_reason_temperature,
        ),
    }
    return JsonLLMClient(backend, tiers, store=store, max_retries=settings.llm_max_retries)
