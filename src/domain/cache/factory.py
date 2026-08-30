import logging

import redis

from src.config import Settings
from src.domain.cache.client import CacheClient, ExactCacheClient
from src.domain.cache.semantic import SemanticCacheClient
from src.domain.jinaai.jina_client import JinaEmbeddingsClient

logger = logging.getLogger(__name__)


def make_redis_client(settings: Settings, *, decode_responses: bool | None = None) -> redis.Redis:
    """Create Redis client with connection pooling."""
    redis_settings = settings.redis
    decode = redis_settings.decode_responses if decode_responses is None else decode_responses

    try:
        client = redis.Redis(
            host=redis_settings.host,
            port=redis_settings.port,
            password=redis_settings.password if redis_settings.password else None,
            db=redis_settings.db,
            decode_responses=decode,
            socket_timeout=redis_settings.socket_timeout,
            socket_connect_timeout=redis_settings.socket_connect_timeout,
            retry_on_timeout=True,
            retry_on_error=[redis.ConnectionError, redis.TimeoutError],
        )

        client.ping()
        logger.info("Connected to Redis at %s:%s (decode_responses=%s)", redis_settings.host, redis_settings.port, decode)
        return client

    except redis.ConnectionError as e:
        logger.error("Failed to connect to Redis: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error creating Redis client: %s", e)
        raise


def _make_semantic_cache_client(settings: Settings) -> SemanticCacheClient | None:
    if not settings.redis.semantic_cache_enabled:
        logger.info("Semantic cache disabled by configuration")
        return None

    try:
        semantic_redis = make_redis_client(settings, decode_responses=False)
        semantic_cache = SemanticCacheClient(semantic_redis, settings.redis)
        logger.info("Semantic cache client created successfully")
        return semantic_cache
    except Exception as e:
        logger.warning("Semantic cache unavailable (Redis Stack with RediSearch required): %s", e)
        return None


def make_cache_client(settings: Settings, embeddings_client: JinaEmbeddingsClient | None = None) -> CacheClient:
    """Create layered exact + semantic cache client for hybrid search."""
    exact_redis = make_redis_client(settings, decode_responses=True)
    exact_cache = ExactCacheClient(exact_redis, settings.redis)
    semantic_cache = _make_semantic_cache_client(settings)

    cache_client = CacheClient(
        exact_cache=exact_cache,
        semantic_cache=semantic_cache,
        embeddings_client=embeddings_client,
        settings=settings.redis,
    )
    logger.info("Hybrid search cache client created successfully")
    return cache_client
