from dataclasses import dataclass


@dataclass
class RetrievalSettings:
    """Mutable retrieval settings updated per request."""

    top_k: int = 3
    use_hybrid: bool = True
    rerank_enabled: bool = True
    rerank_candidate_multiplier: int = 2
    rerank_model: str = "jina-reranker-v2-base-multilingual"
