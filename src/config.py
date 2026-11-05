import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class BaseConfigSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )


class ArxivSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="ARXIV__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    base_url: str = "https://export.arxiv.org/api/query"
    pdf_cache_dir: str = "./data/arxiv_pdfs"
    rate_limit_delay: float = 3.0
    timeout_seconds: int = 30
    max_results: int = 15
    search_category: str = "cs.AI"
    download_max_retries: int = 3
    download_retry_delay_base: float = 5.0
    max_concurrent_downloads: int = 5
    max_concurrent_parsing: int = 1

    namespaces: dict = {
        "atom": "http://www.w3.org/2005/Atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    @field_validator("pdf_cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: str) -> str:
        os.makedirs(v, exist_ok=True)
        return v


class PDFParserSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="PDF_PARSER__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    max_pages: int = 30
    max_file_size_mb: int = 20
    do_ocr: bool = False
    do_table_structure: bool = False
    do_figure_extraction: bool = False
    heading_hierarchy_enabled: bool = True


class ChunkingSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="CHUNKING__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    chunk_size: int = 600  # Target words per chunk
    overlap_size: int = 100  # Words to overlap between chunks
    min_chunk_size: int = 100  # Minimum words for a valid chunk
    section_based: bool = True  # Use section-based chunking when available

    # Contextual Retrieval: LLM-generated context prepended to chunks for search/embedding
    contextualization_enabled: bool = False
    contextualization_model: str = ""  # Empty = use agent_model
    max_document_chars: int = 50_000  # Truncate full document sent to LLM
    max_concurrent_context_requests: int = 3


class OpenSearchSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="OPENSEARCH__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "http://localhost:9200"
    index_name: str = "arxiv-papers"
    chunk_index_suffix: str = "chunks"  # Creates single hybrid index: {index_name}-{suffix}
    # Partition label stamped on indexed chunks; all searches filter by this value.
    ingestion_name: str = "dev"
    max_text_size: int = 1000000

    # Vector search settings
    vector_dimension: int = 1024  # Jina embeddings dimension
    vector_space_type: str = "cosinesimil"  # cosinesimil, l2, innerproduct

    # Hybrid search settings
    rrf_pipeline_name: str = "hybrid-rrf-pipeline"
    hybrid_search_size_multiplier: int = 2  # Get k*multiplier for better recall


class KGSettings(BaseConfigSettings):
    """Feature flag and tuning for LightRAG-style dual-level KG (Postgres graph + OS vectors).

    When ``enabled`` is false, indexing CLI and ``/kg-retrieval/debug`` are inactive.
    ``ingestion_name`` scopes all graph rows and vector docs to the same OpenSearch
    partition as hybrid chunk search (defaults to ``OPENSEARCH__INGESTION_NAME``).
    """

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="KG__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    enabled: bool = False
    # Empty uses OPENSEARCH__INGESTION_NAME at runtime via factory
    ingestion_name: str = ""
    entity_index_suffix: str = "kg-entities"
    relation_index_suffix: str = "kg-relations"
    max_concurrent_extractions: int = 3
    extract_model: str = ""
    max_entities_per_chunk: int = 20
    max_relations_per_chunk: int = 20
    top_k_entities: int = 10
    top_k_relations: int = 10
    max_entity_tokens: int = 4000
    max_relation_tokens: int = 4000
    max_total_tokens: int = 12000
    keyword_model: str = ""
    chunk_rerank_enabled: bool = True
    chunk_rerank_model: str = "jina-reranker-v2-base-multilingual"


class LangfuseSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="LANGFUSE__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    public_key: str = ""
    secret_key: str = ""
    host: str = "https://cloud.langfuse.com"
    enabled: bool = True
    flush_at: int = 15  # Number of events before flushing
    flush_interval: float = 1.0  # Seconds between flushes
    max_retries: int = 3
    timeout: int = 30
    debug: bool = False


class Settings(BaseConfigSettings):
    app_version: str = "0.1.0"
    debug: bool = True
    environment: Literal["development", "staging", "production"] = "development"
    service_name: str = "rag-api"

    postgres_database_url: str = "postgresql://rag_user:rag_password@localhost:5412/rag_db"
    postgres_echo_sql: bool = False
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 0

    openai_api_key: str = ""
    openai_base_url: str | None = None
    # Comma-separated model fallbacks when the primary model fails (e.g. gpt-5.6-luna,gpt-4o-mini)
    llm_fallback_models: str = "gpt-5.6-luna,gpt-4o-mini"
    agent_model: str = "gpt-5.6-luna"
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "low"
    llm_request_timeout: int = 300

    # Jina AI embeddings configuration
    jina_api_key: str = ""

    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    pdf_parser: PDFParserSettings = Field(default_factory=PDFParserSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    kg: KGSettings = Field(default_factory=KGSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)

    @field_validator("postgres_database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not (v.startswith("postgresql://") or v.startswith("postgresql+psycopg2://")):
            raise ValueError("Database URL must start with 'postgresql://' or 'postgresql+psycopg2://'")
        return v


class ConfigurationError(Exception):
    """Exception raised when configuration is invalid."""


def get_settings() -> Settings:
    return Settings()
