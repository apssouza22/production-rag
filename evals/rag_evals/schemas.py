"""Schemas for Ragas RAG evaluation."""

from pydantic import BaseModel, Field


class EvalQuery(BaseModel):
    """A single evaluation query with an optional pre-generated response."""

    user_input: str = Field(min_length=1, description="Question to evaluate")
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


class EvalSample(BaseModel):
    """Evaluation sample with retrieved contexts from rerank search."""

    user_input: str
    response: str
    retrieved_contexts: list[str]
    rerank_applied: bool = False


class MetricScore(BaseModel):
    """Score for a single metric on one sample."""

    name: str
    value: float | None = None
    error: str | None = None
