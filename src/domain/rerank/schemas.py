from typing import Literal

from pydantic import BaseModel, Field


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
