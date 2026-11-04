"""Stable keys and merge helpers shared by indexing and graph upsert."""

import re


def make_chunk_key(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:{chunk_index}"


def relation_vector_id(source_entity_id: str, target_entity_id: str) -> str:
    """Stable OpenSearch document id for a directed relation embedding."""
    return f"{source_entity_id}->{target_entity_id}"


def parse_chunk_key(chunk_key: str) -> tuple[str, int]:
    document_id, index_str = chunk_key.rsplit(":", 1)
    return document_id, int(index_str)


def normalize_entity_id(entity_name: str) -> str:
    """Stable graph key from display name (LightRAG-style slug)."""
    normalized = entity_name.strip().lower()
    normalized = re.sub(r"[^\w\s-]", "", normalized, flags=re.UNICODE)
    normalized = re.sub(r"[\s_-]+", "_", normalized).strip("_")
    return normalized or "unknown_entity"


def merge_descriptions(existing: str, new: str, max_chars: int = 8000) -> str:
    """Append unique sentences from new description; truncate to max_chars."""
    if not existing:
        return new[:max_chars] if new else ""
    if not new or new.strip() in existing:
        return existing[:max_chars]

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", new) if p.strip()]
    merged = existing
    for part in parts:
        if part not in merged:
            merged = f"{merged} {part}".strip()
        if len(merged) >= max_chars:
            break
    return merged[:max_chars]


def merge_unique_strings(existing: list[str], new_items: list[str]) -> list[str]:
    seen = set(existing)
    result = list(existing)
    for item in new_items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
