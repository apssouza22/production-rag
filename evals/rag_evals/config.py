import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    evaluation_llm: str = "gpt-4o-mini"
    evaluation_base_url: str = "https://api.openai.com/v1"
    evaluation_api_key: str = ""
    evaluation_sleep_time: int = 10

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    bifrost_enabled: bool = False
    bifrost_host: str = "http://localhost:8090"
    bifrost_api_key: str = "dummy-key"
    llm_provider: Literal["openai", "bifrost"] = "openai"

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")


def get_settings() -> Settings:
    return Settings()


def build_openai_client_kwargs(settings: Settings | None = None, **overrides) -> dict:
    """Build kwargs for the OpenAI SDK client used by the judge LLM."""
    resolved = settings or get_settings()

    use_bifrost = resolved.bifrost_enabled or resolved.llm_provider == "bifrost"
    if use_bifrost:
        return {
            "api_key": resolved.bifrost_api_key,
            "base_url": f"{resolved.bifrost_host.rstrip('/')}/v1",
            **overrides,
        }

    api_key = (
        overrides.pop("api_key", None)
        or resolved.evaluation_api_key
        or resolved.openai_api_key
        or os.getenv("OPENAI_API_KEY", "")
    )
    base_url = overrides.pop("base_url", None) or resolved.evaluation_base_url
    client_kwargs = {"api_key": api_key, **overrides}
    if base_url:
        client_kwargs["base_url"] = base_url
    return client_kwargs
