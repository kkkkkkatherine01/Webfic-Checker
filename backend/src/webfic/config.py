import uuid
from functools import lru_cache
from typing import Any, Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WEBFIC_",
        env_file=(".env", "../.env"),
        extra="ignore",
    )

    database_url: str = "postgresql+asyncpg://webfic:webfic@localhost:5432/webfic"

    # No login until launch; every record belongs to this user.
    dev_user_id: uuid.UUID = uuid.UUID("00000000-0000-0000-0000-000000000001")

    # Platform LLM provider (DeepSeek by default, via the OpenAI-compatible protocol).
    llm_protocol: Literal["openai"] = "openai"
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: SecretStr | None = None
    llm_extract_model: str = "deepseek-flash"
    llm_reason_model: str = "deepseek-v4-pro"
    # Extra request-body fields per tier, as JSON in .env. The defaults suit DeepSeek:
    # extraction is mechanical, so thinking mode (on by default) only costs tokens.
    llm_extract_extra: dict[str, Any] = {"thinking": {"type": "disabled"}}
    llm_reason_extra: dict[str, Any] = {}
    # Extraction should be as repeatable as possible; None leaves the provider default.
    llm_extract_temperature: float | None = 0.0
    llm_reason_temperature: float | None = None
    llm_max_retries: int = 2

    chunk_size: int = 8000
    chunk_overlap: int = 500


@lru_cache
def get_settings() -> Settings:
    return Settings()
