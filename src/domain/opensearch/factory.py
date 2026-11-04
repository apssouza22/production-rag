"""Unified factory for OpenSearch client and paper RRF search."""

import logging
from functools import lru_cache
from typing import Optional

from src.config import Settings, get_settings

from .client import OpenSearchClient
from .paper_rrf_search import PaperRrfSearch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def make_opensearch_client(settings: Optional[Settings] = None) -> OpenSearchClient:
    """Factory function to create cached OpenSearch transport client."""
    if settings is None:
        settings = get_settings()

    opensearch_client = OpenSearchClient(host=settings.opensearch.host)

    if not opensearch_client.health_check():
        logger.warning("OpenSearch connection failed - search features will be limited")
    else:
        logger.info("OpenSearch connected successfully")

    return opensearch_client


def make_opensearch_client_fresh(settings: Optional[Settings] = None, host: Optional[str] = None) -> OpenSearchClient:
    """Create a fresh OpenSearch client (not cached)."""
    if settings is None:
        settings = get_settings()

    opensearch_host = host or settings.opensearch.host
    return OpenSearchClient(host=opensearch_host)


@lru_cache(maxsize=1)
def make_paper_rrf_search(settings: Optional[Settings] = None) -> PaperRrfSearch:
    """Cached PaperRrfSearch wired to OpenSearch; ensures hybrid index and RRF pipeline."""
    settings = settings or get_settings()

    opensearch_client = make_opensearch_client()
    paper_rrf_search = PaperRrfSearch(opensearch_client, settings)

    if not opensearch_client.health_check():
        return paper_rrf_search

    setup_results = paper_rrf_search.setup_indices(force=False)
    if setup_results.get("hybrid_index"):
        logger.info("Hybrid index created")
    else:
        logger.info("Hybrid index already exists")

    try:
        count = opensearch_client.count(paper_rrf_search.index_name)
        logger.info("OpenSearch ready: %s documents indexed", count)
    except Exception:
        logger.info("OpenSearch index ready (stats unavailable)")

    return paper_rrf_search


def make_paper_rrf_search_fresh(
    settings: Optional[Settings] = None,
    host: Optional[str] = None,
) -> PaperRrfSearch:
    """Fresh PaperRrfSearch instance (e.g. indexing with host override)."""
    if settings is None:
        settings = get_settings()

    opensearch_client = make_opensearch_client_fresh(settings, host=host)
    return PaperRrfSearch(opensearch_client, settings)
