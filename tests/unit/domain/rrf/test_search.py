from unittest.mock import MagicMock

import pytest
from src.config import get_settings
from src.domain.opensearch.client import OpenSearchClient
from src.domain.rrf import HybridRrfSearch


@pytest.fixture
def hybrid_rrf_search() -> HybridRrfSearch:
    opensearch = OpenSearchClient(host="http://localhost:9200")
    search = HybridRrfSearch(opensearch, get_settings())
    search.ingestion_name = "test-partition"
    search._opensearch.client = MagicMock()
    search._opensearch.client.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}
    return search


def test_get_chunks_by_chunk_keys_sorts_by_request_order(hybrid_rrf_search: HybridRrfSearch) -> None:
    hybrid_rrf_search._opensearch.client.search_rrf.return_value = {
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
        ingestion_name="test-partition",
    )
    assert [c["chunk_index"] for c in chunks] == [0, 1]
