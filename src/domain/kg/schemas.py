"""Pydantic models for LLM extraction and retrieval hits."""

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    """One entity record from per-chunk LLM extraction (before merge into ``KgNode``)."""

    entity_name: str = Field(
        min_length=1,
        max_length=256,
        description="Canonical name of a concrete noun or concept stated in the chunk (method, profession, business, metric, architecture, places, etc.).",
    )
    entity_type: str = Field(
        default="UNKNOWN",
        max_length=64,
        description="Short type label (e.g. METHOD, PLACE, METRIC, PROFESSION); use UNKNOWN if unclear.",
    )
    description: str = Field(
        default="",
        max_length=600,
        description="One or two sentences of factual detail from the chunk only; no speculation beyond the text.",
    )


class ExtractedRelation(BaseModel):
    """One directed edge from extraction; endpoints are resolved to ``entity_id`` at upsert time."""

    source_entity: str = Field(
        min_length=1,
        max_length=256,
        description="Head entity name as used in the chunk; must match an entity you would extract.",
    )
    target_entity: str = Field(
        min_length=1,
        max_length=256,
        description="Tail entity name; relationship direction is source → target as stated or implied in the chunk.",
    )
    description: str = Field(
        default="",
        max_length=600,
        description="Brief factual relation (e.g. 'uses', 'outperforms on') grounded in the chunk; 1–2 sentences max.",
    )
    keywords: str = Field(
        default="",
        max_length=300,
        description="Comma-separated or short phrase of retrieval keywords for this edge, taken from the chunk wording.",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Relative strength or confidence of this relation as implied by the chunk (default 1.0).",
    )


class ChunkExtractionResult(BaseModel):
    """Structured LLM output for a single OpenSearch text chunk."""

    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description=(
            "Entities explicitly supported by the chunk text; server may truncate to the configured "
            "max after validation. Omit entities not grounded in this chunk."
        ),
    )
    relations: list[ExtractedRelation] = Field(
        default_factory=list,
        description=(
            "Directed edges between extracted entities; both endpoints must appear as entity names "
            "in this chunk. Server may truncate to the configured max after validation."
        ),
    )


class DualKeywords(BaseModel):
    """Query understanding split: themes (global/relations) vs specifics (local/entities)."""

    high_level_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Broad themes and abstract concepts (task, problem setting, paradigms) for graph/relation-oriented retrieval."
        ),
    )
    low_level_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Specific surface terms: named methods, models, datasets, metrics, numbers, or proper nouns for entity/chunk retrieval."
        ),
    )


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


class KgPathHit(BaseModel):
    """Multi-hop entity path with PathRAG-style narrative relation context."""

    entity_ids: list[str] = Field(default_factory=list)
    hop_count: int = 0
    weight: float = 0.0
    narrative: str = ""


class KgChunkHit(BaseModel):
    """Evidence text loaded from the hybrid chunk index via ``source_chunk_keys`` on nodes/edges."""

    chunk_key: str
    arxiv_id: str
    chunk_index: int
    chunk_text: str
    title: str = ""
    authors: str = ""
    section_name: str = ""
    rerank_score: float | None = None


class DualLevelKGResult(BaseModel):
    """Full debug/eval payload from ``DualLevelKGRetrievalService.retrieve``."""

    query: str
    high_level_keywords: list[str] = Field(default_factory=list)
    low_level_keywords: list[str] = Field(default_factory=list)
    entities: list[KgEntityHit] = Field(default_factory=list)
    relations: list[KgRelationHit] = Field(default_factory=list)
    paths: list[KgPathHit] = Field(default_factory=list)
    chunks: list[KgChunkHit] = Field(default_factory=list)
    formatted_context: str = ""
