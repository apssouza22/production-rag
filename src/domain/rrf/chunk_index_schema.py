"""Generic hybrid chunk index schema: strict core fields + dynamic metadata object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from src.config import OpenSearchSettings

# Denormalized paper / document fields stored under metadata (not strict top-level).
METADATA_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "title",
        "authors",
        "abstract",
        "published_date",
        "pdf_url",
        "chunk_word_count",
        "start_char",
        "end_char",
        "section_title",
        "section_breadcrumb",
    }
)

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
    "arxiv_id": {"type": "keyword"},
    "paper_id": {"type": "keyword"},
    "chunk_index": {"type": "integer"},
    "chunk_text": TEXT_FIELD,
    "chunk_context": {"type": "text", "analyzer": "text_analyzer"},
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
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_context: str | None = None
    arxiv_id: str | None = None
    paper_id: str | None = None
    embedding_model: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    model_config = {"extra": "forbid"}

    @classmethod
    def from_index_payload(cls, payload: dict[str, Any]) -> ChunkIndexDocument:
        """Normalize flat or nested payloads into core + metadata."""
        data = payload.copy()
        metadata = dict(data.pop("metadata", None) or {})
        for name in METADATA_FIELD_NAMES:
            if name in data:
                metadata[name] = data.pop(name)
        if "categories" in metadata and "categories" not in data:
            data["categories"] = metadata.pop("categories")
        return cls.model_validate({**data, "metadata": metadata})

    def to_opensearch_source(self, *, embedding: list[float] | None = None) -> dict[str, Any]:
        source: dict[str, Any] = self.model_dump(exclude_none=True)
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
    paper_level_bm25_fields: list[str]
    paper_level_source_fields: list[str]
    paper_level_highlight_fields: dict[str, dict[str, Any]]


_DEFAULT_HIGHLIGHT_CHUNK = {
    "fragment_size": 150,
    "number_of_fragments": 2,
    "pre_tags": ["<mark>"],
    "post_tags": ["</mark>"],
}

PAPER_CHUNK_PROFILE = ChunkSearchProfile(
    bm25_fields=[
        "contextualized_text^4",
        "chunk_text^2"
    ],
    term_filters={"categories": "categories"},
    sort_field="metadata.published_date",
    highlight_fields={
        "contextualized_text": _DEFAULT_HIGHLIGHT_CHUNK,
        "chunk_text": _DEFAULT_HIGHLIGHT_CHUNK,
        "metadata.title": {"fragment_size": 0, "number_of_fragments": 0, "pre_tags": ["<mark>"], "post_tags": ["</mark>"]},
        "metadata.abstract": {
            "fragment_size": 150,
            "number_of_fragments": 1,
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
        },
    },
    paper_level_bm25_fields=["metadata.title^3", "metadata.abstract^2", "metadata.authors^1"],
    paper_level_source_fields=[
        "document_id",
        "arxiv_id",
        "metadata.title",
        "metadata.authors",
        "metadata.abstract",
        "categories",
        "metadata.published_date",
        "metadata.pdf_url",
    ],
    paper_level_highlight_fields={
        "metadata.title": {"fragment_size": 0, "number_of_fragments": 0},
        "metadata.abstract": {
            "fragment_size": 150,
            "number_of_fragments": 3,
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
        },
        "metadata.authors": {
            "fragment_size": 0,
            "number_of_fragments": 0,
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
        },
    },
)
