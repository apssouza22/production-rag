from pydantic import BaseModel, Field

from src.domain.kg.schemas import DualLevelKGResult, KgChunkHit, KgEntityHit, KgRelationHit


class KgRetrievalDebugRequest(BaseModel):
    """Input for ``POST /api/v1/kg-retrieval/debug`` (does not run the LangGraph agent)."""

    query: str = Field(min_length=1)
    top_k_entities: int | None = Field(default=None, ge=1, le=50)
    top_k_relations: int | None = Field(default=None, ge=1, le=50)


class KgRetrievalDebugResponse(DualLevelKGResult):
    """Same shape as ``DualLevelKGResult``; exposed for OpenAPI and eval inspection."""


__all__ = [
    "KgRetrievalDebugRequest",
    "KgRetrievalDebugResponse",
    "KgEntityHit",
    "KgRelationHit",
    "KgChunkHit",
]
