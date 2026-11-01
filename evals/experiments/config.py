from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = REPO_ROOT / ".env"
DEFAULT_PDFS_DIR = Path(__file__).parent / "arxiv_pdfs"
DEFAULT_QUERIES_PATH = Path(__file__).parent / "generated_queries.json"


class ExperimentSettings(BaseSettings):
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

    # OpenSearch partition the experiment indexes into and searches. Also exported as
    # OPENSEARCH__INGESTION_NAME so retrieval filters match what was indexed.
    experiment_ingestion_name: str = "evals-experiments"
    experiment_pdfs_dir: Path = DEFAULT_PDFS_DIR
    experiment_queries_path: Path = DEFAULT_QUERIES_PATH
    experiment_questions_per_paper: int = Field(default=4, ge=1, le=20)
    experiment_chunks_per_paper: int = Field(default=12, ge=1, le=100)
    experiment_top_k: int = Field(default=5, ge=1, le=50)

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")

    @property
    def judge_api_key(self) -> str:
        return self.evaluation_api_key or self.openai_api_key


def get_settings() -> ExperimentSettings:
    return ExperimentSettings()
