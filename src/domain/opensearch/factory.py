"""Unified factory for OpenSearch transport client."""

import logging
from functools import lru_cache
from typing import Optional

from src.config import Settings, get_settings

from .client import OpenSearchClient

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
