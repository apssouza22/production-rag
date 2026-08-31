from typing import Optional

from src.config import get_settings
from src.domain.jinaai.jina_client import JinaEmbeddingsClient
from src.domain.jinaai.jina_reranker_client import JinaRerankerClient
from src.platform.langfuse.client import LangfuseTracer
from src.platform.llm.protocol import LlmProviderClient
from src.domain.opensearch.client import OpenSearchClient

from .agentic_rag import AgenticRAGService
from .config import GraphConfig
from .graph import AgenticRAGGraph
from .retrieval_settings import RetrievalSettings


def make_agentic_rag_graph(
    llm_client: LlmProviderClient,
    opensearch_client: OpenSearchClient,
    embeddings_client: JinaEmbeddingsClient,
    graph_config: GraphConfig,
    reranker_client: JinaRerankerClient | None = None,
    retrieval_settings: RetrievalSettings | None = None,
    langfuse_tracer: Optional[LangfuseTracer] = None,
) -> tuple[AgenticRAGGraph, RetrievalSettings]:
    """Build AgenticRAGGraph and its shared retrieval settings."""
    settings = retrieval_settings or RetrievalSettings(
        top_k=graph_config.top_k,
        use_hybrid=graph_config.use_hybrid,
        rerank_enabled=graph_config.rerank_enabled,
        rerank_candidate_multiplier=graph_config.rerank_candidate_multiplier,
        rerank_model=graph_config.rerank_model,
    )
    graph_builder = AgenticRAGGraph(
        llm_client=llm_client,
        opensearch_client=opensearch_client,
        embeddings_client=embeddings_client,
        retrieval_settings=settings,
        config=graph_config,
        reranker_client=reranker_client,
        langfuse_tracer=langfuse_tracer,
    )
    return graph_builder, settings


def make_agentic_rag_service(
    opensearch_client: OpenSearchClient,
    llm_client: LlmProviderClient,
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
        rerank_candidate_multiplier=settings.opensearch.hybrid_search_size_multiplier,
    )
    graph_builder, retrieval_settings = make_agentic_rag_graph(
        llm_client=llm_client,
        opensearch_client=opensearch_client,
        embeddings_client=embeddings_client,
        graph_config=graph_config,
        reranker_client=reranker_client,
        langfuse_tracer=langfuse_tracer,
    )

    return AgenticRAGService(
        llm_client=llm_client,
        graph_builder=graph_builder,
        retrieval_settings=retrieval_settings,
        reranker_client=reranker_client,
        langfuse_tracer=langfuse_tracer,
        graph_config=graph_config,
    )
