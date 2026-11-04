from src.domain.jinaai.jina_client import JinaEmbeddingsClient
from src.domain.jinaai.jina_reranker_client import JinaRerankerClient
from src.domain.rrf import PaperRrfSearch

from .config import RerankSearchConfig
from .service import SearchService


def make_rerank_search_service(
    paper_rrf_search: PaperRrfSearch,
    embeddings_client: JinaEmbeddingsClient,
    config: RerankSearchConfig | None = None,
    reranker_client: JinaRerankerClient | None = None,
) -> SearchService:
    """Factory function to create a rerank search service.

    :param paper_rrf_search: Paper chunk hybrid RRF search for retrieval
    :param embeddings_client: Jina embeddings client for query vectors
    :param config: Retrieval and rerank configuration
    :param reranker_client: Optional Jina reranker client
    :returns: Configured SearchService instance
    """
    return SearchService(
        paper_rrf_search=paper_rrf_search,
        embeddings_client=embeddings_client,
        config=config or RerankSearchConfig(),
        reranker_client=reranker_client,
    )
