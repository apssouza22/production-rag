from typing import Optional

from src.config import get_settings
from src.domain.jinaai.jina_client import JinaEmbeddingsClient
from src.domain.jinaai.jina_reranker_client import JinaRerankerClient
from src.domain.opensearch.client import OpenSearchClient
from src.domain.rerank.config import RerankSearchConfig
from src.domain.rerank.factory import make_rerank_search_service
from src.domain.rerank.service import RerankSearchService
from src.platform.langfuse.client import LangfuseTracer
from src.platform.llm.protocol import LlmProviderClient

from .agentic_rag import AgenticRAGService
from .config import GraphConfig
from .graph import AgenticRAGGraph


def _build_rerank_search_config(graph_config: GraphConfig) -> RerankSearchConfig:
    return RerankSearchConfig(
        use_hybrid=graph_config.use_hybrid,
        rerank_enabled=graph_config.rerank_enabled,
        rerank_model=graph_config.rerank_model,
    )


def make_agentic_rag_graph(
    llm_client: LlmProviderClient,
    graph_config: GraphConfig,
    rerank_search_service: RerankSearchService,
    langfuse_tracer: Optional[LangfuseTracer] = None,
) -> AgenticRAGGraph:
    """Build AgenticRAGGraph wired to a rerank search service."""
    return AgenticRAGGraph(
        llm_client=llm_client,
        rerank_search_service=rerank_search_service,
        config=graph_config,
        langfuse_tracer=langfuse_tracer,
    )


def make_agentic_rag_service(
    llm_client: LlmProviderClient,
    opensearch_client: OpenSearchClient,
    embeddings_client: JinaEmbeddingsClient,
    reranker_client: JinaRerankerClient | None = None,
    langfuse_tracer: Optional[LangfuseTracer] = None,
    top_k: int = 3,
    use_hybrid: bool = True,
) -> AgenticRAGService:
    """Create AgenticRAGService with dependency injection."""
    settings = get_settings()
    graph_config = GraphConfig(
        top_k=top_k,
        use_hybrid=use_hybrid,
        model=settings.agent_model,
    )
    rerank_search_service = make_rerank_search_service(
        opensearch_client=opensearch_client,
        embeddings_client=embeddings_client,
        reranker_client=reranker_client,
        config=_build_rerank_search_config(graph_config),
    )
    graph_builder = make_agentic_rag_graph(
        llm_client=llm_client,
        graph_config=graph_config,
        rerank_search_service=rerank_search_service,
        langfuse_tracer=langfuse_tracer,
    )

    return AgenticRAGService(
        llm_client=llm_client,
        graph_builder=graph_builder,
        rerank_search_service=rerank_search_service,
        reranker_client=reranker_client,
        langfuse_tracer=langfuse_tracer,
        graph_config=graph_config,
    )
