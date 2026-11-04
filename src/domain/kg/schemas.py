"""Pydantic models for LLM extraction and retrieval hits."""

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    """One entity record from per-chunk LLM extraction (before merge into ``KgNode``)."""

    entity_name: str
    entity_type: str = "UNKNOWN"
    description: str = ""


class ExtractedRelation(BaseModel):
    """One directed edge from extraction; endpoints are resolved to ``entity_id`` at upsert time."""

    source_entity: str
    target_entity: str
    description: str = ""
    keywords: str = ""
    weight: float = 1.0


class ChunkExtractionResult(BaseModel):
    """Structured LLM output for a single OpenSearch text chunk."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


class DualKeywords(BaseModel):
    """Query understanding split: themes (global/relations) vs specifics (local/entities)."""

    high_level_keywords: list[str] = Field(default_factory=list)
    low_level_keywords: list[str] = Field(default_factory=list)


class KgEntityHit(BaseModel):
    """Entity row returned to callers after vector search + graph hydrate; ``rank`` is node degree."""

    entity_id: str
    entity_name: str
    entity_type: str
    description: str
    rank: int = 0
    source_chunk_keys: list[str] = Field(default_factory=list)


class KgRelationHit(BaseModel):
    """Relation row after relation-vector search; ``rank`` reflects stored edge weight."""

    source_entity_id: str
    target_entity_id: str
    description: str
    keywords: str
    weight: float
    rank: int = 0
    source_chunk_keys: list[str] = Field(default_factory=list)


class KgChunkHit(BaseModel):
    """Evidence text loaded from the hybrid chunk index via ``source_chunk_keys`` on nodes/edges."""

    chunk_key: str
    arxiv_id: str
    chunk_index: int
    chunk_text: str
    title: str = ""
    authors: str = ""
    section_name: str = ""


class DualLevelKGResult(BaseModel):
    """Full debug/eval payload from ``DualLevelKGRetrievalService.retrieve``."""

    query: str
    high_level_keywords: list[str] = Field(default_factory=list)
    low_level_keywords: list[str] = Field(default_factory=list)
    entities: list[KgEntityHit] = Field(default_factory=list)
    relations: list[KgRelationHit] = Field(default_factory=list)
    chunks: list[KgChunkHit] = Field(default_factory=list)
    formatted_context: str = ""
