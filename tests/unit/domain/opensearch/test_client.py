from unittest.mock import MagicMock

import pytest
from src.domain.opensearch.client import OpenSearchClient


@pytest.fixture
def client() -> OpenSearchClient:
    opensearch_client = OpenSearchClient(host="http://localhost:9200")
    opensearch_client.client = MagicMock()
    opensearch_client.client.cluster.health.return_value = {"status": "green"}
    return opensearch_client


def test_health_check_returns_true_when_cluster_green(client: OpenSearchClient) -> None:
    assert client.health_check() is True


def test_ensure_index_mapping_updates_existing_index(client: OpenSearchClient) -> None:
    client.client.indices.exists.return_value = True
    mapping_body = {"mappings": {"properties": {"document_id": {"type": "keyword"}}}}

    client.ensure_index_mapping("arxiv-papers-chunks", mapping_body)

    client.client.indices.put_mapping.assert_called_once_with(
        index="arxiv-papers-chunks",
        body={"properties": {"document_id": {"type": "keyword"}}},
    )
