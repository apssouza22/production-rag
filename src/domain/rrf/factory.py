"""Factory for hybrid RRF chunk search wired to OpenSearch."""

import logging
from functools import lru_cache
from typing import Optional

from src.config import Settings, get_settings
from src.domain.opensearch.client import OpenSearchClient
from src.domain.opensearch.factory import make_opensearch_client, make_opensearch_client_fresh

from .search import HybridRrfSearch, PaperRrfSearch

logger = logging.getLogger(__name__)


def _setup_hybrid_rrf_search(hybrid_rrf_search: HybridRrfSearch, opensearch_client: OpenSearchClient) -> HybridRrfSearch:
    if not opensearch_client.health_check():
        return hybrid_rrf_search

    setup_results = hybrid_rrf_search.setup_indices(force=False)
    if setup_results.get("hybrid_index"):
        logger.info("Hybrid index created")
    else:
        logger.info("Hybrid index already exists")

    try:
        count = opensearch_client.count(hybrid_rrf_search.index_name)
        logger.info("OpenSearch ready: %s documents indexed", count)
    except Exception:
        logger.info("OpenSearch index ready (stats unavailable)")

    return hybrid_rrf_search


@lru_cache(maxsize=16)
def make_hybrid_rrf_search(
    settings: Optional[Settings] = None,
    ingestion_name: str | None = None,
) -> HybridRrfSearch:
    """Cached HybridRrfSearch wired to OpenSearch; ensures hybrid index and RRF pipeline."""
    settings = settings or get_settings()

    opensearch_client = make_opensearch_client()
    hybrid_rrf_search = HybridRrfSearch(opensearch_client, settings, ingestion_name=ingestion_name)
    return _setup_hybrid_rrf_search(hybrid_rrf_search, opensearch_client)


def make_hybrid_rrf_search_fresh(
    settings: Optional[Settings] = None,
    host: Optional[str] = None,
    ingestion_name: str | None = None,
) -> HybridRrfSearch:
    """Fresh HybridRrfSearch instance (e.g. indexing with host override)."""
    if settings is None:
        settings = get_settings()

    opensearch_client = make_opensearch_client_fresh(settings, host=host)
    return HybridRrfSearch(opensearch_client, settings, ingestion_name=ingestion_name)


def make_paper_rrf_search(
    settings: Optional[Settings] = None,
    ingestion_name: str | None = None,
) -> PaperRrfSearch:
    """Backward-compatible alias for make_hybrid_rrf_search."""
    return make_hybrid_rrf_search(settings, ingestion_name=ingestion_name)


def make_paper_rrf_search_fresh(
    settings: Optional[Settings] = None,
    host: Optional[str] = None,
    ingestion_name: str | None = None,
) -> PaperRrfSearch:
    """Backward-compatible alias for make_hybrid_rrf_search_fresh."""
    return make_hybrid_rrf_search_fresh(settings, host=host, ingestion_name=ingestion_name)
