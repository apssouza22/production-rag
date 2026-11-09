from unittest.mock import MagicMock

import pytest
from src.config import get_settings
from src.domain.opensearch.client import OpenSearchClient
from src.domain.rrf import HybridRrfSearch
from src.domain.rrf.indexing import MAX_CHUNKS_PER_DOCUMENT_QUERY
from src.domain.rrf.parent_expansion import ParentSectionExpander


@pytest.fixture
def hybrid_rrf_search() -> HybridRrfSearch:
    opensearch = OpenSearchClient(host="http://localhost:9200")
    search = HybridRrfSearch(opensearch, get_settings(), ingestion_name="test-partition")
    search._opensearch.client = MagicMock()
    return search


def test_get_chunks_by_section_queries_core_breadcrumb(hybrid_rrf_search: HybridRrfSearch) -> None:
    hybrid_rrf_search._opensearch.client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "os-0",
                    "_source": {
                        "document_id": "2401.00001",
                        "chunk_index": 0,
                        "section_breadcrumb": "Methods > Training",
                        "chunk_text": "first",
                    },
                }
            ]
        }
    }

    expander: ParentSectionExpander = hybrid_rrf_search.parent_section_expander
    chunks = expander.get_chunks_by_section("2401.00001", "Methods > Training")
    assert len(chunks) == 1
    assert chunks[0]["section_breadcrumb"] == "Methods > Training"

    body = hybrid_rrf_search._opensearch.client.search.call_args.kwargs["body"]
    must = body["query"]["bool"]["must"]
    assert {"term": {"document_id": "2401.00001"}} in must
    assert {"term": {"section_breadcrumb": "Methods > Training"}} in must
    assert {"term": {"ingestion_name": "test-partition"}} in must
    assert body["size"] == MAX_CHUNKS_PER_DOCUMENT_QUERY
