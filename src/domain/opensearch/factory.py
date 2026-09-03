"""Unified factory for OpenSearch client."""
import logging
from functools import lru_cache
from typing import Optional

from src.config import Settings, get_settings

from .client import OpenSearchClient

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def make_opensearch_client(settings: Optional[Settings] = None) -> OpenSearchClient:
    """Factory function to create cached OpenSearch client.

    Uses lru_cache to maintain a singleton instance for efficiency.

    :param settings: Optional settings instance
    :returns: Cached OpenSearchClient instance
    """
    if settings is None:
        settings = get_settings()

    opensearch_client =  OpenSearchClient(host=settings.opensearch.host, settings=settings)

    if not opensearch_client.health_check():
        logger.warning("OpenSearch connection failed - search features will be limited")
        return opensearch_client

    logger.info("OpenSearch connected successfully")
    setup_results = opensearch_client.setup_indices(force=False)
    if setup_results.get("hybrid_index"):
        logger.info("Hybrid index created")
    else:
        logger.info("Hybrid index already exists")

    try:
        stats = opensearch_client.client.count(index=opensearch_client.index_name)
        logger.info(f"OpenSearch ready: {stats['count']} documents indexed")
    except Exception:
        logger.info("OpenSearch index ready (stats unavailable)")
    return opensearch_client


def make_opensearch_client_fresh(settings: Optional[Settings] = None, host: Optional[str] = None) -> OpenSearchClient:
    """Factory function to create a fresh OpenSearch client (not cached).

    Use this when you need a new client instance (e.g., for testing
    or when connection issues occur).

    :param settings: Optional settings instance
    :param host: Optional host override
    :returns: New OpenSearchClient instance
    """
    if settings is None:
        settings = get_settings()

    # Use provided host or settings host
    opensearch_host = host or settings.opensearch.host

    return OpenSearchClient(host=opensearch_host, settings=settings)
