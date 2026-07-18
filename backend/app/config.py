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

    # Comma-separated browser origins allowed to call the API directly (CORS). The
    # production frontend proxies /api through Vercel, so it is same-origin and does not
    # need CORS — but we keep this configurable for direct-call setups.
    allowed_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "https://medical-hackathon-livid.vercel.app"
    )

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
