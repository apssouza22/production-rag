from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import KGSettings
from src.domain.kg.dual_level_retrieval import DualLevelKGRetrievalService
from src.domain.kg.models import KgNode
from src.domain.kg.schemas import DualKeywords


@pytest.fixture
def kg_settings() -> KGSettings:
    return KGSettings(
        enabled=True,
        top_k_entities=5,
        top_k_relations=5,
        max_entity_tokens=500,
        max_relation_tokens=500,
        max_total_tokens=2000,
    )


@pytest.mark.asyncio
async def test_retrieve_builds_context_from_entity_branch(kg_settings: KGSettings) -> None:
    extraction = AsyncMock()
    extraction.extract_keywords.return_value = DualKeywords(
        high_level_keywords=[],
        low_level_keywords=["transformer"],
    )

    embeddings = AsyncMock()
    embeddings.embed_query.return_value = [0.1] * 8

    kg_vector = MagicMock()
    kg_vector.knn_search_entities.return_value = [{"entity_id": "transformer", "score": 1.0}]
    kg_vector.knn_search_relations.return_value = []

    node = KgNode(
        entity_id="transformer",
        entity_name="Transformer",
        entity_type="MODEL",
        description="Attention-based model",
        source_chunk_keys=["1234.5678:0"],
        arxiv_ids=["1234.5678"],
        ingestion_name="dev",
    )

    session = MagicMock()
    database = MagicMock()
    database.get_session.return_value.__enter__ = MagicMock(return_value=session)
    database.get_session.return_value.__exit__ = MagicMock(return_value=None)

    repo_patch = MagicMock()
    repo_patch.get_nodes_by_entity_ids.return_value = [node]
    repo_patch.node_degrees.return_value = {"transformer": 2}

    opensearch = MagicMock()
    opensearch.get_chunks_by_chunk_keys.return_value = [
        {
            "chunk_key": "1234.5678:0",
            "arxiv_id": "1234.5678",
            "chunk_index": 0,
            "chunk_text": "We propose the Transformer.",
            "title": "Attention",
            "authors": "Vaswani",
            "section_title": "Intro",
        }
    ]

    service = DualLevelKGRetrievalService(
        database=database,
        opensearch_client=opensearch,
        kg_vector_client=kg_vector,
        extraction_service=extraction,
        embeddings_client=embeddings,
        kg_settings=kg_settings,
        ingestion_name="dev",
    )

    with patch("src.domain.kg.dual_level_retrieval.KgGraphRepository", return_value=repo_patch):
        result = await service.retrieve("What is a transformer?")

    assert result.low_level_keywords == ["transformer"]
    assert len(result.entities) == 1
    assert len(result.chunks) == 1
    assert "Transformer" in result.formatted_context
