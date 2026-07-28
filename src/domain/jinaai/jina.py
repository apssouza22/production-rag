from typing import Dict, List

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class JinaEmbeddingRequest(BaseModel):
    """Request model for Jina embeddings API."""

    model: str = "jina-embeddings-v3"
    task: str = "retrieval.passage"  # or "retrieval.query" for queries
    dimensions: int = 1024
    late_chunking: bool = False
    embedding_type: str = "float"
    input: List[str]


class JinaEmbeddingResponse(BaseModel):
    """Response model from Jina embeddings API."""

    model: str
    object: str = "list"
    usage: Dict[str, int]
    data: List[Dict]


class JinaRerankRequest(BaseModel):
    """Request model for Jina reranker API."""

    model: str = "jina-reranker-v2-base-multilingual"
    query: str
    documents: List[str]
    top_n: int


class JinaRerankResult(BaseModel):
    """Single reranked document from Jina reranker API."""

    model_config = ConfigDict(populate_by_name=True)

    index: int
    relevance_score: float = Field(validation_alias=AliasChoices("relevance_score", "score"))
    document: Dict[str, str] | None = None


class JinaRerankResponse(BaseModel):
    """Response model from Jina reranker API."""

    model: str
    results: List[JinaRerankResult]
    usage: Dict[str, int] | None = None
