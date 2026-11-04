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
