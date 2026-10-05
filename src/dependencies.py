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

from src.agents.fusionsearch.agentic_rag import AgenticRAGService
from src.agents.knowledgerouter import KnowledgeRouterService
from src.agents.texttosql import TextToSQLService
from src.config import Settings
from src.domain.arxiv.client import ArxivClient
from src.domain.db.interfaces.base import BaseDatabase
from src.domain.jinaai.jina_client import JinaEmbeddingsClient
from src.domain.jinaai.jina_reranker_client import JinaRerankerClient
from src.domain.opensearch.client import OpenSearchClient
from src.domain.pdf_parser.parser import PDFParserService
from src.domain.rerank.service import RerankSearchService
from src.platform.cache.client import CacheClient
from src.platform.langfuse.client import LangfuseTracer


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


def get_rerank_search_service(request: Request) -> RerankSearchService:
    """Get rerank search service from the request state."""
    return request.app.state.rerank_search_service


def get_langfuse_tracer(request: Request) -> LangfuseTracer:
    """Get Langfuse tracer from the request state."""
    return request.app.state.langfuse_tracer


def get_cache_client(request: Request) -> CacheClient | None:
    """Get cache client from the request state."""
    return getattr(request.app.state, "cache_client", None)


def get_agentic_rag_service(request: Request) -> AgenticRAGService:
    """Get agentic RAG service from the request state."""
    return request.app.state.agentic_rag_service


def get_text_to_sql_service(request: Request) -> TextToSQLService:
    """Get text-to-SQL service from the request state."""
    return request.app.state.text_to_sql_service


def get_knowledge_router_service(request: Request) -> KnowledgeRouterService:
    """Get knowledge router service from the request state."""
    return request.app.state.knowledge_router_service


SettingsDep = Annotated[Settings, Depends(get_request_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
OpenSearchDep = Annotated[OpenSearchClient, Depends(get_opensearch_client)]
ArxivDep = Annotated[ArxivClient, Depends(get_arxiv_client)]
PDFParserDep = Annotated[PDFParserService, Depends(get_pdf_parser)]
EmbeddingsDep = Annotated[JinaEmbeddingsClient, Depends(get_embeddings_service)]
RerankerDep = Annotated[JinaRerankerClient, Depends(get_reranker_service)]
RerankSearchDep = Annotated[RerankSearchService, Depends(get_rerank_search_service)]
LangfuseDep = Annotated[LangfuseTracer, Depends(get_langfuse_tracer)]
CacheDep = Annotated[CacheClient | None, Depends(get_cache_client)]
AgenticRAGDep = Annotated[AgenticRAGService, Depends(get_agentic_rag_service)]
TextToSQLDep = Annotated[TextToSQLService, Depends(get_text_to_sql_service)]
KnowledgeRouterDep = Annotated[KnowledgeRouterService, Depends(get_knowledge_router_service)]
