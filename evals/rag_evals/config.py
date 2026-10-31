from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = REPO_ROOT / ".env"
DEFAULT_DATASET_PATH = Path(__file__).parent / "queries.json"


class RagEvalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        case_sensitive=False,
        populate_by_name=True,
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

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    rag_eval_experiment_name: str = "rag-retrieval-eval"
    rag_eval_experiment_description: str = "Ragas retrieval evaluation via rerank search"
    rag_eval_langfuse_dataset: str = ""
    rag_eval_max_concurrency: int = Field(default=3, ge=1, le=50)

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")


def get_settings() -> RagEvalSettings:
    return RagEvalSettings()
