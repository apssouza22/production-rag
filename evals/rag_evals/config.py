from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from custom_evals.config import build_openai_client_kwargs

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"
DEFAULT_DATASET_PATH = Path(__file__).parent / "queries.json"


class RagEvalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    evaluation_llm: str = "gpt-4o-mini"
    evaluation_base_url: str = "https://api.openai.com/v1"
    evaluation_api_key: str = ""

    rag_api_base_url: str = "http://localhost:8000"
    rag_eval_dataset_path: Path = DEFAULT_DATASET_PATH
    rag_eval_top_k: int = Field(default=5, ge=1, le=50)
    rag_eval_use_hybrid: bool = True
    rag_eval_rerank_enabled: bool = True
    rag_eval_generate_responses: bool = True
    rag_eval_request_timeout: float = 120.0

    bifrost_enabled: bool = False
    bifrost_host: str = "http://localhost:8090"
    bifrost_api_key: str = "sk-bf-agent-1-dev"
    llm_provider: str = "openai"

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")


def get_settings() -> RagEvalSettings:
    return RagEvalSettings()


def build_eval_openai_client_kwargs(settings: RagEvalSettings | None = None) -> dict:
    """Build kwargs for the OpenAI SDK client used by Ragas metrics."""
    resolved = settings or get_settings()
    use_bifrost = resolved.bifrost_enabled or resolved.llm_provider == "bifrost"
    if use_bifrost:
        return build_openai_client_kwargs(
            api_key=resolved.bifrost_api_key,
            base_url=f"{resolved.bifrost_host.rstrip('/')}/v1",
        )

    return build_openai_client_kwargs(
        api_key=resolved.evaluation_api_key or resolved.openai_api_key or None,
        base_url=resolved.evaluation_base_url or None,
    )
