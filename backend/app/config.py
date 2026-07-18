from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables (and an optional .env)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # When set, lifestyle / side-effect concerns are extracted with an LLM classifier
    # (more robust to free-text phrasing). When empty, the system falls back to the
    # deterministic keyword rules, so the app works fully offline with no API key.
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    # Base URL is overridable for proxies / Azure-style gateways.
    openai_base_url: str = "https://api.openai.com/v1"
    # Hard timeout (seconds) for the classifier call; on timeout we fall back to keywords.
    llm_timeout_seconds: float = 8.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
