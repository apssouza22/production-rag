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
    assert "arxiv_id" not in props
    assert "paper_id" not in props


def test_chunk_index_document_accepts_arbitrary_metadata() -> None:
    doc = ChunkIndexDocument.from_index_payload(
        {
            "document_id": "2024.1",
            "chunk_index": 0,
            "chunk_text": "body",
            "contextualized_text": "ctx\n\nbody",
            "ingestion_name": "dev",
            "categories": ["cs.AI"],
            "metadata": {
                "title": "Paper",
                "section_title": "Intro",
                "start_char": 10,
                "custom_tag": "eval-run-1",
            },
        }
    )
    assert doc.metadata["title"] == "Paper"
    assert doc.metadata["custom_tag"] == "eval-run-1"
    assert doc.categories == ["cs.AI"]
    source = doc.to_opensearch_source()
    assert "title" not in source
    assert source["metadata"]["custom_tag"] == "eval-run-1"
    assert source["categories"] == ["cs.AI"]
    assert "created_at" in source
    assert "updated_at" in source


def test_to_opensearch_source_preserves_created_at_and_refreshes_updated_at() -> None:
    doc = ChunkIndexDocument(
        document_id="1",
        chunk_index=0,
        chunk_text="t",
        contextualized_text="t",
        ingestion_name="dev",
        created_at="2020-01-01T00:00:00+00:00",
    )
    source = doc.to_opensearch_source()
    assert source["created_at"] == "2020-01-01T00:00:00+00:00"
    assert source["updated_at"] != "2020-01-01T00:00:00+00:00"
    assert source["updated_at"].endswith("+00:00")


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


def test_chunk_index_document_rejects_denormalized_fields_at_top_level() -> None:
    with pytest.raises(ValidationError):
        ChunkIndexDocument.from_index_payload(
            {
                "document_id": "2024.1",
                "chunk_index": 0,
                "chunk_text": "body",
                "contextualized_text": "ctx",
                "ingestion_name": "dev",
                "title": "Paper",
            }
        )


def test_flatten_hit_metadata() -> None:
    flat = flatten_hit_metadata(
        {
            "document_id": "1",
            "metadata": {"title": "T", "abstract": "A", "arxiv_id": "1"},
        }
    )
    assert flat["title"] == "T"
    assert flat["abstract"] == "A"
    assert flat["arxiv_id"] == "1"
    assert "metadata" not in flat
