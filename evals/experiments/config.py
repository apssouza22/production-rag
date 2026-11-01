import json
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.config import ChunkingSettings, PDFParserSettings
from src.domain.arxiv_ingestion.common import get_cached_services
from src.domain.pdf_parser.factory import make_pdf_parser_service
from src.domain.rerank.config import RerankSearchConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = REPO_ROOT / ".env"
DEFAULT_PDFS_DIR = Path(__file__).parent / "arxiv_pdfs"
DEFAULT_QUERIES_PATH = Path(__file__).parent / "generated_queries.json"
DEFAULT_EXPERIMENT_CONFIG_PATH = Path(__file__).parent / "data/settings/experiment-config.json"

_SECRET_EXPERIMENT_FIELDS = frozenset({"openai_api_key", "evaluation_api_key"})


class ExperimentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        case_sensitive=False,
        populate_by_name=True,
    )

    evaluation_llm: str = "gpt-5.6-luna"
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


class QuerySettings(BaseModel):
    """OpenSearch / rerank retrieval knobs for comparing experiment runs."""

    model_config = ConfigDict(frozen=True)

    use_hybrid: bool = True
    rerank_enabled: bool = True
    rerank_model: str = "jina-reranker-v2-base-multilingual"
    rerank_candidate_multiplier: int = Field(default=2, ge=1, le=10)

    def to_rerank_config(self) -> RerankSearchConfig:
        return RerankSearchConfig(
            use_hybrid=self.use_hybrid,
            rerank_enabled=self.rerank_enabled,
            rerank_model=self.rerank_model,
            rerank_candidate_multiplier=self.rerank_candidate_multiplier,
        )


class ExperimentConfigFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    pdf_parser: PDFParserSettings
    chunking: ChunkingSettings
    query: QuerySettings
    experiment: ExperimentSettings


@dataclass(frozen=True)
class ExperimentRunConfig:
    pdf_parser: PDFParserSettings
    chunking: ChunkingSettings
    query: QuerySettings
    experiment: ExperimentSettings
    config_path: Path


def resolve_settings_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def _serialize_env_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _apply_prefixed_env(prefix: str, settings: BaseSettings) -> None:
    for key, value in settings.model_dump(mode="json").items():
        os.environ[f"{prefix}__{key.upper()}"] = _serialize_env_value(value)


def _apply_experiment_env(experiment: ExperimentSettings) -> None:
    for key, value in experiment.model_dump(mode="json").items():
        if key in _SECRET_EXPERIMENT_FIELDS:
            continue
        env_name = key.upper()
        if env_name in os.environ:
            continue
        os.environ[env_name] = _serialize_env_value(value)


def apply_experiment_platform_settings(pdf_parser: PDFParserSettings, chunking: ChunkingSettings) -> None:
    """Push parser/chunking config into env and reset cached ingestion services."""
    _apply_prefixed_env("PDF_PARSER", pdf_parser)
    _apply_prefixed_env("CHUNKING", chunking)
    make_pdf_parser_service.cache_clear()
    get_cached_services.cache_clear()


def _settings_defaults(settings_cls: type[BaseSettings]) -> dict[str, object]:
    return settings_cls.model_construct().model_dump()


def load_experiment_config(path: Path | None = None) -> ExperimentConfigFile:
    config_path = path or DEFAULT_EXPERIMENT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment settings file not found: {config_path}")

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in experiment settings file {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Experiment settings file must contain a JSON object: {config_path}")

    pdf_section = raw.get("pdf_parser", {})
    chunk_section = raw.get("chunking", {})
    query_section = raw.get("query", {})
    experiment_section = raw.get("experiment", {})

    if not isinstance(pdf_section, dict):
        raise ValueError("'pdf_parser' must be a JSON object")
    if not isinstance(chunk_section, dict):
        raise ValueError("'chunking' must be a JSON object")
    if not isinstance(query_section, dict):
        raise ValueError("'query' must be a JSON object")
    if not isinstance(experiment_section, dict):
        raise ValueError("'experiment' must be a JSON object")

    return ExperimentConfigFile(
        pdf_parser=PDFParserSettings.model_validate({**_settings_defaults(PDFParserSettings), **pdf_section}),
        chunking=ChunkingSettings.model_validate({**_settings_defaults(ChunkingSettings), **chunk_section}),
        query=QuerySettings.model_validate(query_section),
        experiment=ExperimentSettings.model_validate({**_settings_defaults(ExperimentSettings), **experiment_section}),
    )


def build_run_config(config_path: Path | None = None) -> ExperimentRunConfig:
    """Load JSON config and merge with environment (defaults → JSON → env)."""
    resolved = resolve_settings_path(config_path or DEFAULT_EXPERIMENT_CONFIG_PATH)
    file_config = load_experiment_config(resolved)
    apply_experiment_platform_settings(file_config.pdf_parser, file_config.chunking)
    _apply_experiment_env(file_config.experiment)
    # Environment (including .env) wins over JSON for experiment fields and secrets.
    experiment = ExperimentSettings()
    return ExperimentRunConfig(
        pdf_parser=file_config.pdf_parser,
        chunking=file_config.chunking,
        query=file_config.query,
        experiment=experiment,
        config_path=resolved,
    )


def get_settings(config_path: Path | None = None) -> ExperimentRunConfig:
    return build_run_config(config_path)
