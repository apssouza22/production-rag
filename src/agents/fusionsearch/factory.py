from typing import Optional

from src.config import get_settings
from src.domain.jinaai.jina_client import JinaEmbeddingsClient
from src.domain.jinaai.jina_reranker_client import JinaRerankerClient
from src.domain.opensearch.client import OpenSearchClient
from src.domain.rerank.factory import make_rerank_search_service
from src.domain.rerank.service import RerankSearchService
from src.platform.langfuse.client import LangfuseTracer
from src.platform.llm.protocol import LlmProviderClient

from .agentic_rag import AgenticRAGService
from .config import GraphConfig
from .graph import AgenticRAGGraph
from .retrieval_settings import RetrievalSettings


def make_agentic_rag_graph(
    llm_client: LlmProviderClient,
    graph_config: GraphConfig,
    rerank_search_service: RerankSearchService | None = None,
    retrieval_settings: RetrievalSettings | None = None,
    langfuse_tracer: Optional[LangfuseTracer] = None,
    *,
    opensearch_client: OpenSearchClient | None = None,
    embeddings_client: JinaEmbeddingsClient | None = None,
    reranker_client: JinaRerankerClient | None = None,
) -> tuple[AgenticRAGGraph, RetrievalSettings]:
    """Build AgenticRAGGraph and its shared retrieval settings."""
    settings = retrieval_settings or RetrievalSettings(
        top_k=graph_config.top_k,
        use_hybrid=graph_config.use_hybrid,
        rerank_enabled=graph_config.rerank_enabled,
        rerank_candidate_multiplier=graph_config.rerank_candidate_multiplier,
        rerank_model=graph_config.rerank_model,
    )
    resolved_rerank_search_service = rerank_search_service
    if resolved_rerank_search_service is None:
        if opensearch_client is None or embeddings_client is None:
            raise ValueError("rerank_search_service or opensearch/embeddings clients are required")
        resolved_rerank_search_service = make_rerank_search_service(
            opensearch_client=opensearch_client,
            embeddings_client=embeddings_client,
            reranker_client=reranker_client,
        )

    graph_builder = AgenticRAGGraph(
        llm_client=llm_client,
        rerank_search_service=resolved_rerank_search_service,
        retrieval_settings=settings,
        config=graph_config,
        langfuse_tracer=langfuse_tracer,
    )
    return graph_builder, settings


def make_agentic_rag_service(
    llm_client: LlmProviderClient,
    rerank_search_service: RerankSearchService | None = None,
    reranker_client: JinaRerankerClient | None = None,
    langfuse_tracer: Optional[LangfuseTracer] = None,
    top_k: int = 3,
    use_hybrid: bool = True,
    *,
    opensearch_client: OpenSearchClient | None = None,
    embeddings_client: JinaEmbeddingsClient | None = None,
) -> AgenticRAGService:
    """Create AgenticRAGService with dependency injection."""
    settings = get_settings()
    graph_config = GraphConfig(
        top_k=top_k,
        use_hybrid=use_hybrid,
        model=settings.agent_model,
        rerank_candidate_multiplier=settings.opensearch.hybrid_search_size_multiplier,
    )
    graph_builder, retrieval_settings = make_agentic_rag_graph(
        llm_client=llm_client,
        graph_config=graph_config,
        rerank_search_service=rerank_search_service,
        langfuse_tracer=langfuse_tracer,
        opensearch_client=opensearch_client,
        embeddings_client=embeddings_client,
        reranker_client=reranker_client,
    )

    return AgenticRAGService(
        llm_client=llm_client,
        graph_builder=graph_builder,
        retrieval_settings=retrieval_settings,
        reranker_client=reranker_client,
        langfuse_tracer=langfuse_tracer,
        graph_config=graph_config,
    )
