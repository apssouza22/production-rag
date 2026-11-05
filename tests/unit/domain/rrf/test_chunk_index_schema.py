import pytest
from pydantic import ValidationError
from src.config import get_settings
from src.domain.rrf.chunk_index_schema import (
    ChunkIndexDocument,
    build_hybrid_chunk_mapping,
    flatten_hit_metadata,
)


def test_build_hybrid_chunk_mapping_has_strict_core_and_dynamic_metadata() -> None:
    mapping = build_hybrid_chunk_mapping(get_settings().opensearch)
    assert mapping["mappings"]["dynamic"] == "strict"
    props = mapping["mappings"]["properties"]
    assert props["document_id"]["type"] == "keyword"
    assert props["categories"]["type"] == "keyword"
    assert props["metadata"]["dynamic"] is True
    assert "title" not in props


def test_chunk_index_document_moves_paper_fields_to_metadata() -> None:
    doc = ChunkIndexDocument.from_index_payload(
        {
            "document_id": "2024.1",
            "chunk_index": 0,
            "chunk_text": "body",
            "contextualized_text": "ctx\n\nbody",
            "ingestion_name": "dev",
            "title": "Paper",
            "categories": ["cs.AI"],
            "section_title": "Intro",
            "start_char": 10,
        }
    )
    assert doc.metadata["title"] == "Paper"
    assert doc.categories == ["cs.AI"]
    assert "categories" not in doc.metadata
    assert doc.metadata["section_title"] == "Intro"
    assert doc.metadata["start_char"] == 10
    source = doc.to_opensearch_source()
    assert "title" not in source
    assert "section_title" not in source
    assert source["metadata"]["section_title"] == "Intro"
    assert source["categories"] == ["cs.AI"]


def test_chunk_index_document_lifts_categories_from_legacy_metadata() -> None:
    doc = ChunkIndexDocument.from_index_payload(
        {
            "document_id": "2024.1",
            "chunk_index": 0,
            "chunk_text": "body",
            "contextualized_text": "ctx\n\nbody",
            "ingestion_name": "dev",
            "metadata": {"categories": ["cs.LG"], "title": "Paper"},
        }
    )
    assert doc.categories == ["cs.LG"]
    assert "categories" not in doc.metadata
    assert doc.metadata["title"] == "Paper"


def test_chunk_index_document_rejects_unknown_top_level_fields() -> None:
    with pytest.raises(ValidationError):
        ChunkIndexDocument.from_index_payload(
            {
                "document_id": "x",
                "chunk_index": 0,
                "chunk_text": "t",
                "contextualized_text": "t",
                "ingestion_name": "dev",
                "unknown_field": "bad",
            }
        )


def test_flatten_hit_metadata() -> None:
    flat = flatten_hit_metadata(
        {
            "document_id": "1",
            "metadata": {"title": "T", "abstract": "A"},
        }
    )
    assert flat["title"] == "T"
    assert flat["abstract"] == "A"
    assert "metadata" not in flat
