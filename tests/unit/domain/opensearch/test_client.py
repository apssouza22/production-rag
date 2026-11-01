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


def test_hybrid_search_keeps_hybrid_query_at_top_level(client: OpenSearchClient):
    """OpenSearch rejects a hybrid query nested inside another query."""
    client.search_chunks_hybrid(query="transformers", query_embedding=[0.1] * 4, size=5)

    search_body = client.client.search.call_args.kwargs["body"]
    assert set(search_body["query"]) == {"hybrid"}


def test_hybrid_search_filters_both_subqueries_by_ingestion_name(client: OpenSearchClient):
    """Both sides of the fusion must honour the ingestion partition filter."""
    client.search_chunks_hybrid(query="transformers", query_embedding=[0.1] * 4, size=5)

    search_body = client.client.search.call_args.kwargs["body"]
    bm25_query, embedding_query = search_body["query"]["hybrid"]["queries"]
    ingestion_filter = {"term": {"ingestion_name": "test-partition"}}

    assert ingestion_filter in bm25_query["bool"]["filter"]
    assert ingestion_filter in embedding_query["bool"]["filter"]
    assert "knn" in embedding_query["bool"]["must"][0]
