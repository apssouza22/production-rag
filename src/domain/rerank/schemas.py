from typing import Literal

from pydantic import BaseModel, Field


class RerankSearchRequest(BaseModel):
    """Request model for OpenSearch retrieval with optional Jina reranking."""

    query: str = Field(
        default="machine learning neural networks",
        min_length=1,
        max_length=500,
        description="Search query text",
    )
    top_k: int = Field(default=10, ge=1, le=50, description="Number of documents to return after reranking")
    use_hybrid: bool = Field(default=True, description="Enable hybrid search (BM25 + vector)")
    rerank_enabled: bool = Field(default=True, description="Rerank retrieved candidates with Jina")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "machine learning neural networks",
                "top_k": 10,
                "use_hybrid": True,
                "rerank_enabled": True,
            }
        }


class SearchDocument(BaseModel):
    """OpenSearch hit normalized for retrieval and reranking."""

    arxiv_id: str
    chunk_text: str
    title: str = ""
    authors: str = ""
    section_name: str = ""
    score: float = 0.0
    rank: int = Field(description="Zero-based position in the result list")
    chunk_id: str | None = None
    rerank_score: float | None = None
    original_rank: int | None = Field(
        default=None,
        description="Zero-based rank before reranking, when reranking was applied",
    )


class RerankSearchResult(BaseModel):
    """OpenSearch retrieval result with optional Jina reranking."""

    query: str
    search_mode: Literal["hybrid", "bm25"]
    rerank_applied: bool
    before_rerank: list[SearchDocument]
    after_rerank: list[SearchDocument]
