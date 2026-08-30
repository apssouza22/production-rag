from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, Generator

if TYPE_CHECKING:
    from fastapi import Depends, Request
    from sqlalchemy.orm import Session
else:
    try:
        from fastapi import Depends, Request
        from sqlalchemy.orm import Session
    except ImportError:
        pass

from src.config import Settings
from src.agents.fusionsearch.agentic_rag import AgenticRAGService
from src.agents.fusionsearch.factory import make_agentic_rag_service
from src.agents.knowledgerouter import KnowledgeRouterService, make_knowledge_router_service
from src.agents.texttosql import TextToSQLService, make_text_to_sql_service
from src.domain.arxiv.client import ArxivClient
from src.domain.cache.client import CacheClient
from src.domain.db.interfaces.base import BaseDatabase
from src.domain.jinaai.jina_client import JinaEmbeddingsClient
from src.domain.jinaai.jina_reranker_client import JinaRerankerClient
from src.domain.langfuse.client import LangfuseTracer
from src.domain.llm.factory import make_agent_llm_client
from src.domain.llm.protocol import RagClient
from src.domain.opensearch.client import OpenSearchClient
from src.domain.pdf_parser.parser import PDFParserService


@lru_cache
def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


def get_request_settings(request: Request) -> Settings:
    """Get settings from the request state."""
    return request.app.state.settings


def get_database(request: Request) -> BaseDatabase:
    """Get database from the request state."""
    return request.app.state.database


def get_db_session(database: Annotated[BaseDatabase, Depends(get_database)]) -> Generator[Session, None, None]:
    """Get database session dependency."""
    with database.get_session() as session:
        yield session


def get_opensearch_client(request: Request) -> OpenSearchClient:
    """Get OpenSearch client from the request state."""
    return request.app.state.opensearch_client


def get_arxiv_client(request: Request) -> ArxivClient:
    """Get arXiv client from the request state."""
    return request.app.state.arxiv_client


def get_pdf_parser(request: Request) -> PDFParserService:
    """Get PDF parser service from the request state."""
    return request.app.state.pdf_parser


def get_embeddings_service(request: Request) -> JinaEmbeddingsClient:
    """Get embeddings service from the request state."""
    return request.app.state.embeddings_service


def get_reranker_service(request: Request) -> JinaRerankerClient:
    """Get reranker service from the request state."""
    return request.app.state.reranker_service


def get_llm_client(request: Request) -> RagClient:
    """Get the configured LLM client from the request state."""
    return request.app.state.llm_client


def get_ollama_client(request: Request) -> RagClient:
    """Backward-compatible alias for get_llm_client."""
    return get_llm_client(request)


def get_langfuse_tracer(request: Request) -> LangfuseTracer:
    """Get Langfuse tracer from the request state."""
    return request.app.state.langfuse_tracer


def get_cache_client(request: Request) -> CacheClient | None:
    """Get cache client from the request state."""
    return getattr(request.app.state, "cache_client", None)


# Dependency annotations
SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
OpenSearchDep = Annotated[OpenSearchClient, Depends(get_opensearch_client)]
ArxivDep = Annotated[ArxivClient, Depends(get_arxiv_client)]
PDFParserDep = Annotated[PDFParserService, Depends(get_pdf_parser)]
EmbeddingsDep = Annotated[JinaEmbeddingsClient, Depends(get_embeddings_service)]
RerankerDep = Annotated[JinaRerankerClient, Depends(get_reranker_service)]
LLMDep = Annotated[RagClient, Depends(get_llm_client)]
OllamaDep = Annotated[RagClient, Depends(get_llm_client)]
LangfuseDep = Annotated[LangfuseTracer, Depends(get_langfuse_tracer)]
CacheDep = Annotated[CacheClient | None, Depends(get_cache_client)]


def get_agentic_rag_service(
    opensearch: OpenSearchDep,
    embeddings: EmbeddingsDep,
    reranker: RerankerDep,
    langfuse: LangfuseDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AgenticRAGService:
    """Get agentic RAG service (Bifrost virtual key: agent-1)."""
    return make_agentic_rag_service(
        opensearch_client=opensearch,
        llm_client=make_agent_llm_client("agent_1", settings),
        embeddings_client=embeddings,
        reranker_client=reranker,
        langfuse_tracer=langfuse,
    )


AgenticRAGDep = Annotated[AgenticRAGService, Depends(get_agentic_rag_service)]


def get_text_to_sql_service(
    langfuse: LangfuseDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> TextToSQLService:
    """Get text-to-SQL service (Bifrost virtual key: agent-2)."""
    return make_text_to_sql_service(
        llm_client=make_agent_llm_client("agent_2", settings),
        langfuse_tracer=langfuse,
        model=settings.agent_model,
    )


TextToSQLDep = Annotated[TextToSQLService, Depends(get_text_to_sql_service)]


def get_knowledge_router_service(
    agentic_rag: AgenticRAGDep,
    text_to_sql: TextToSQLDep,
    langfuse: LangfuseDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> KnowledgeRouterService:
    """Get knowledge router service (Bifrost virtual key: agent-2)."""
    return make_knowledge_router_service(
        agentic_rag_service=agentic_rag,
        text_to_sql_service=text_to_sql,
        llm_client=make_agent_llm_client("agent_2", settings),
        langfuse_tracer=langfuse,
        model=settings.agent_model,
    )


KnowledgeRouterDep = Annotated[KnowledgeRouterService, Depends(get_knowledge_router_service)]
