"""Pydantic models for LLM extraction, retrieval hits, and prompt context assembly."""

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


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token) for budget trimming."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def format_kg_context(
    entities: list[KgEntityHit],
    relations: list[KgRelationHit],
    chunks: list[KgChunkHit],
    *,
    max_entity_tokens: int,
    max_relation_tokens: int,
    max_total_tokens: int,
) -> str:
    """Build LLM-ready context: CSV-like entity/relation blocks plus delimited chunk text."""
    entity_lines = ["id,entity_name,entity_type,description,rank"]
    for idx, entity in enumerate(entities, start=1):
        entity_lines.append(
            f'{idx},"{entity.entity_name}","{entity.entity_type}","{entity.description.replace(chr(34), chr(39))}",{entity.rank}'
        )
    relation_lines = ["id,source,target,description,keywords,weight,rank"]
    for idx, relation in enumerate(relations, start=1):
        relation_lines.append(
            f'{idx},"{relation.source_entity_id}","{relation.target_entity_id}",'
            f'"{relation.description.replace(chr(34), chr(39))}","{relation.keywords}",{relation.weight},{relation.rank}'
        )

    entities_block = truncate_text_to_tokens("\n".join(entity_lines), max_entity_tokens)
    relations_block = truncate_text_to_tokens("\n".join(relation_lines), max_relation_tokens)

    chunk_parts: list[str] = []
    used = estimate_tokens(entities_block) + estimate_tokens(relations_block)
    chunk_budget = max(0, max_total_tokens - used)
    for chunk in chunks:
        header = f"[{chunk.arxiv_id} chunk {chunk.chunk_index}] {chunk.title}"
        if chunk.section_name:
            header += f" — {chunk.section_name}"
        body = f"{header}\n{chunk.chunk_text}"
        if estimate_tokens("\n\n".join(chunk_parts + [body])) > chunk_budget:
            break
        chunk_parts.append(body)

    sections = []
    if entities_block:
        sections.append(f"--- Entities (KG) ---\n{entities_block}")
    if relations_block:
        sections.append(f"--- Relations (KG) ---\n{relations_block}")
    if chunk_parts:
        sections.append(f"--- Evidence chunks ---\n\n--New Chunk--\n".join(chunk_parts))
    return "\n\n".join(sections)
