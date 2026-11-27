from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = REPO_ROOT / ".env"
DEFAULT_DATASET_PATH = Path(__file__).parent / "queries.json"


class TextToSQLEvalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        case_sensitive=False,
        populate_by_name=True,
    )

    text_to_sql_eval_dataset_path: Path = DEFAULT_DATASET_PATH
    text_to_sql_eval_item_delay_seconds: float = Field(default=0.0, ge=0.0)
    text_to_sql_eval_model: str = ""
    text_to_sql_eval_user_id: str = "texttosql_eval"


def get_settings() -> TextToSQLEvalSettings:
    return TextToSQLEvalSettings()