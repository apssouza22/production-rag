"""Schemas for Ragas RAG evaluation."""

from pydantic import BaseModel, Field

from evals.core.schemas import EvalSample, MetricResult


class EvalQuery(BaseModel):
    """A single evaluation query with a reference answer and optional response."""

    user_input: str = Field(min_length=1, description="Question to evaluate")
    reference: str = Field(min_length=1, description="Ground-truth answer for reference-based metrics")
    response: str | None = Field(default=None, description="Optional answer used by response-based metrics")


class RerankSearchDocument(BaseModel):
    """Document returned by the rerank search API."""

    chunk_text: str
    arxiv_id: str = ""
    title: str = ""
    rank: int = 0


class RerankSearchResponse(BaseModel):
    """Response payload from /api/v1/rerank-search."""

    query: str
    rerank_applied: bool
    after_rerank: list[RerankSearchDocument]


MetricScore = MetricResult

__all__ = [
    "EvalQuery",
    "EvalSample",
    "MetricScore",
    "RerankSearchDocument",
    "RerankSearchResponse",
]
