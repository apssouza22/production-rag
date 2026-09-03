from src.domain.jinaai.jina_client import JinaEmbeddingsClient
from src.domain.jinaai.jina_reranker_client import JinaRerankerClient
from src.domain.opensearch.client import OpenSearchClient

from .config import RerankSearchConfig
from .service import RerankSearchService


def make_rerank_search_service(
    opensearch_client: OpenSearchClient,
    embeddings_client: JinaEmbeddingsClient,
    config: RerankSearchConfig | None = None,
    reranker_client: JinaRerankerClient | None = None,
) -> RerankSearchService:
    """Factory function to create a rerank search service.

    :param opensearch_client: OpenSearch client for retrieval
    :param embeddings_client: Jina embeddings client for query vectors
    :param config: Retrieval and rerank configuration
    :param reranker_client: Optional Jina reranker client
    :returns: Configured RerankSearchService instance
    """
    return RerankSearchService(
        opensearch_client=opensearch_client,
        embeddings_client=embeddings_client,
        config=config or RerankSearchConfig(),
        reranker_client=reranker_client,
    )
