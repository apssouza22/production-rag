from dataclasses import dataclass


@dataclass
class RerankSearchConfig:
    """Mutable reranking settings for RerankSearchService."""

    use_hybrid: bool = True
    rerank_enabled: bool = True
    rerank_model: str = "jina-reranker-v2-base-multilingual"
