from dataclasses import dataclass


@dataclass
class RerankSearchConfig:
    """Mutable reranking settings for SearchService."""

    use_hybrid: bool = True
    rerank_enabled: bool = True
    rerank_model: str = "jina-reranker-v2-base-multilingual"
    rerank_candidate_multiplier: int = 2
