from unittest.mock import MagicMock

import pytest
from src.config import get_settings
from src.domain.opensearch.client import OpenSearchClient
from src.domain.rrf import HybridRrfSearch


@pytest.fixture
def hybrid_rrf_search() -> HybridRrfSearch:
    opensearch = OpenSearchClient(host="http://localhost:9200")
    search = HybridRrfSearch(opensearch, get_settings(), ingestion_name="test-partition")
    search._opensearch.client = MagicMock()
    search._opensearch.client.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}
    return search


def test_get_chunks_by_chunk_keys_sorts_by_request_order(hybrid_rrf_search: HybridRrfSearch) -> None:
    hybrid_rrf_search._opensearch.client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "os-2",
                    "_source": {
                        "document_id": "1234.5678",
                        "chunk_index": 1,
                        "chunk_text": "second",
                    },
                },
                {
                    "_id": "os-1",
                    "_source": {
                        "document_id": "1234.5678",
                        "chunk_index": 0,
                        "chunk_text": "first",
                    },
                },
            ]
        }
    }
    chunks = hybrid_rrf_search.get_chunks_by_chunk_keys(
        ["1234.5678:0", "1234.5678:1"],
    )
    assert [c["chunk_index"] for c in chunks] == [0, 1]


def test_search_unified_expand_parent_sections_calls_expansion(hybrid_rrf_search: HybridRrfSearch) -> None:
    hybrid_rrf_search._search_bm25_only = MagicMock(
        return_value={
            "total": 1,
            "hits": [
                {
                    "document_id": "2401.00001",
                    "chunk_index": 0,
                    "section_breadcrumb": "Methods > Training",
                    "chunk_text": "child",
                }
            ],
        }
    )
    hybrid_rrf_search._parent_section_expander.expand = MagicMock(
        return_value=[{"chunk_text": "parent", "parent_expansion_applied": True}]
    )

    result = hybrid_rrf_search.search_unified(
        query="test",
        query_embedding=None,
        use_hybrid=False,
        expand_parent_sections=True,
    )

    hybrid_rrf_search._parent_section_expander.expand.assert_called_once()
    assert result["hits"][0]["parent_expansion_applied"] is True


def test_setup_indices_updates_mapping_when_index_already_exists(hybrid_rrf_search: HybridRrfSearch) -> None:
    hybrid_rrf_search._opensearch.create_index = MagicMock(return_value=False)
    hybrid_rrf_search._opensearch.ensure_index_mapping = MagicMock()
    hybrid_rrf_search._opensearch.ensure_search_pipeline = MagicMock(return_value=True)

    hybrid_rrf_search.setup_indices(force=False)

    hybrid_rrf_search._opensearch.ensure_index_mapping.assert_called_once()
