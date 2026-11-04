"""Format dual-level KG retrieval hits into LLM-ready context strings."""

from src.domain.kg.schemas import KgChunkHit, KgEntityHit, KgRelationHit


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
        sections.append(f"--- Evidence chunks ---\n".join(chunk_parts))
    return "\n\n".join(sections)
