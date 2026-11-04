from unittest.mock import MagicMock

import pytest
from src.config import get_settings
from src.domain.opensearch.client import OpenSearchClient


@pytest.fixture
def client() -> OpenSearchClient:
    opensearch_client = OpenSearchClient(host="http://localhost:9200", settings=get_settings())
    opensearch_client.ingestion_name = "test-partition"
    opensearch_client.client = MagicMock()
    opensearch_client.client.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}
    return opensearch_client


def test_get_chunks_by_chunk_keys_sorts_by_request_order(client: OpenSearchClient) -> None:
    client.client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "os-2",
                    "_source": {
                        "arxiv_id": "1234.5678",
                        "chunk_index": 1,
                        "chunk_text": "second",
                    },
                },
                {
                    "_id": "os-1",
                    "_source": {
                        "arxiv_id": "1234.5678",
                        "chunk_index": 0,
                        "chunk_text": "first",
                    },
                },
            ]
        }
    }
    chunks = client.get_chunks_by_chunk_keys(["1234.5678:0", "1234.5678:1"], ingestion_name="test-partition")
    assert [c["chunk_index"] for c in chunks] == [0, 1]

