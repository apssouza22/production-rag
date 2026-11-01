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

