"""Generic hybrid chunk index schema: strict core fields + dynamic metadata object."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from src.config import OpenSearchSettings

TEXT_FIELD = {
    "type": "text",
    "analyzer": "text_analyzer",
    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
}


def _embedding_field(vector_dimension: int, space_type: str) -> dict[str, Any]:
    return {
        "type": "knn_vector",
        "dimension": vector_dimension,
        "method": {
            "name": "hnsw",
            "space_type": space_type,
            "engine": "nmslib",
            "parameters": {"ef_construction": 512, "m": 16},
        },
    }


CORE_CHUNK_PROPERTIES: dict[str, Any] = {
    "chunk_id": {"type": "keyword"},
    "document_id": {"type": "keyword"},
    "chunk_index": {"type": "integer"},
    "section_breadcrumb": {"type": "keyword"},
    "chunk_text": TEXT_FIELD,
    "contextualized_text": TEXT_FIELD,
    "embedding_model": {"type": "keyword"},
    "ingestion_name": {"type": "keyword"},
    "categories": {"type": "keyword"},
    "created_at": {"type": "date"},
    "updated_at": {"type": "date"},
    "metadata": {"type": "object", "dynamic": True},
}


def build_hybrid_chunk_mapping(settings: OpenSearchSettings) -> dict[str, Any]:
    """Build OpenSearch index body for hybrid BM25 + kNN chunk search."""
    properties = dict(CORE_CHUNK_PROPERTIES)
    properties["embedding"] = _embedding_field(
        settings.vector_dimension,
        settings.vector_space_type,
    )
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "index.knn": True,
            "index.knn.space_type": settings.vector_space_type,
            "analysis": {
                "analyzer": {
                    "standard_analyzer": {"type": "standard", "stopwords": "_english_"},
                    "text_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "stop", "snowball"],
                    },
                }
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": properties,
        },
    }


def flatten_hit_metadata(hit: dict[str, Any]) -> dict[str, Any]:
    """Merge nested metadata into the hit for legacy consumers expecting top-level fields."""
    result = dict(hit)
    metadata = result.pop("metadata", None)
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if key not in result:
                result[key] = value
    return result


class ChunkIndexDocument(BaseModel):
    """Validated chunk document for OpenSearch indexing (core + metadata)."""

    document_id: str
    chunk_index: int
    chunk_text: str
    contextualized_text: str
    ingestion_name: str
    categories: list[str] = Field(default_factory=list)
    section_breadcrumb: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    model_config = {"extra": "forbid"}

    @classmethod
    def from_index_payload(cls, payload: dict[str, Any]) -> ChunkIndexDocument:
        """Validate core fields and an optional metadata object (any keys allowed under metadata)."""
        data = payload.copy()
        metadata = data.pop("metadata", None)
        if metadata is None:
            metadata = {}
        else:
            metadata = dict(metadata)
            metadata.pop("section_breadcrumb", None)
        return cls.model_validate({**data, "metadata": metadata})

    def to_opensearch_source(self, *, embedding: list[float] | None = None) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        source: dict[str, Any] = self.model_dump(exclude_none=True)
        if source.get("created_at") is None:
            source["created_at"] = now
        source["updated_at"] = now
        if embedding is not None:
            source["embedding"] = embedding
        return source


@dataclass(frozen=True)
class ChunkSearchProfile:
    """BM25, filter, sort, and highlight configuration for chunk search."""

    bm25_fields: list[str]
    term_filters: dict[str, str]
    sort_field: str
    highlight_fields: dict[str, dict[str, Any]]


_DEFAULT_HIGHLIGHT_CHUNK = {
    "fragment_size": 150,
    "number_of_fragments": 2,
    "pre_tags": ["<mark>"],
    "post_tags": ["</mark>"],
}

DEFAULT_CHUNK_PROFILE = ChunkSearchProfile(
    bm25_fields=[
        "contextualized_text^4",
        "chunk_text^2",
    ],
    term_filters={"categories": "categories"},
    sort_field="updated_at",
    highlight_fields={
        "contextualized_text": _DEFAULT_HIGHLIGHT_CHUNK,
        "chunk_text": _DEFAULT_HIGHLIGHT_CHUNK,
    },
)
