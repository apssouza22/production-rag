"""``make_*`` factories wiring KG services to Postgres, OpenSearch, Jina, and LLM settings."""

from src.config import Settings, get_settings
from src.domain.db.factory import make_database
from src.domain.db.interfaces.base import BaseDatabase
from src.domain.jinaai.factory import make_embeddings_service, make_reranker_client
from src.domain.jinaai.jina_reranker_client import JinaRerankerClient
from src.domain.kg.dual_level_retrieval import DualLevelKGRetrievalService
from src.domain.kg.extraction import KgExtractionService
from src.domain.kg.indexing_service import KgIndexingService
from src.domain.kg.kg_search_client import KgSearchClient
from src.domain.kg.path_retrieval import PathDiscover
from src.domain.opensearch.client import OpenSearchClient
from src.domain.opensearch.factory import make_opensearch_client
from src.domain.rrf.factory import make_hybrid_rrf_search
from src.platform.llm.chat_model import make_chat_model


def resolve_kg_ingestion_name(settings: Settings) -> str:
    if settings.kg.ingestion_name.strip():
        return settings.kg.ingestion_name.strip()
    return settings.opensearch.ingestion_name


def make_kg_search_client(
    opensearch_client: OpenSearchClient | None = None,
    settings: Settings | None = None,
    ingestion_name: str | None = None,
) -> KgSearchClient:
    settings = settings or get_settings()
    client = opensearch_client or make_opensearch_client()
    resolved_ingestion = ingestion_name or resolve_kg_ingestion_name(settings)
    return KgSearchClient(
        client=client.client,
        settings=settings,
        ingestion_name=resolved_ingestion,
    )


def make_kg_extraction_service(settings: Settings | None = None) -> KgExtractionService:
    settings = settings or get_settings()
    extract_model = settings.kg.extract_model.strip() or settings.agent_model
    chat_model = make_chat_model(settings, extract_model, temperature=0.0)
    return KgExtractionService(
        chat_model,
        max_entities_per_chunk=settings.kg.max_entities_per_chunk,
        max_relations_per_chunk=settings.kg.max_relations_per_chunk,
        max_concurrent=settings.kg.max_concurrent_extractions,
    )


def make_keyword_extraction_service(settings: Settings | None = None) -> KgExtractionService:
    settings = settings or get_settings()
    keyword_model = settings.kg.keyword_model.strip() or settings.agent_model
    chat_model = make_chat_model(settings, keyword_model, temperature=0.0)
    return KgExtractionService(
        chat_model,
        max_entities_per_chunk=settings.kg.max_entities_per_chunk,
        max_relations_per_chunk=settings.kg.max_relations_per_chunk,
        max_concurrent=settings.kg.max_concurrent_extractions,
    )


def make_kg_indexing_service(
    database: BaseDatabase | None = None,
    settings: Settings | None = None,
) -> KgIndexingService:
    settings = settings or get_settings()
    database = database or make_database()
    opensearch_client = make_opensearch_client()
    ingestion_name = resolve_kg_ingestion_name(settings)
    paper_rrf_search = make_hybrid_rrf_search(ingestion_name=ingestion_name)
    return KgIndexingService(
        database=database,
        paper_rrf_search=paper_rrf_search,
        kg_search_client=make_kg_search_client(opensearch_client, settings, ingestion_name),
        extraction_service=make_kg_extraction_service(settings),
        embeddings_client=make_embeddings_service(),
        ingestion_name=ingestion_name,
    )


def make_path_discover(settings: Settings | None = None) -> PathDiscover:
    settings = settings or get_settings()
    kg = settings.kg
    return PathDiscover(
        enabled=kg.path_retrieval_enabled,
        max_hops=kg.path_max_hops,
        max_paths=kg.path_max_paths,
        weight_threshold=kg.path_weight_threshold,
        weight_alpha=kg.path_weight_alpha,
    )


def make_dual_level_kg_retrieval_service(
    database: BaseDatabase | None = None,
    settings: Settings | None = None,
    reranker_client: JinaRerankerClient | None = None,
) -> DualLevelKGRetrievalService:
    settings = settings or get_settings()
    database = database or make_database()
    opensearch_client = make_opensearch_client()
    ingestion_name = resolve_kg_ingestion_name(settings)
    paper_rrf_search = make_hybrid_rrf_search(ingestion_name=ingestion_name)
    return DualLevelKGRetrievalService(
        database=database,
        paper_rrf_search=paper_rrf_search,
        kg_search_client=make_kg_search_client(opensearch_client, settings, ingestion_name),
        extraction_service=make_keyword_extraction_service(settings),
        embeddings_client=make_embeddings_service(),
        kg_settings=settings.kg,
        ingestion_name=ingestion_name,
        reranker_client=reranker_client if reranker_client is not None else make_reranker_client(settings),
        path_discover=make_path_discover(settings),
    )
