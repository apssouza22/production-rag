import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal, Optional

import redis
from src.config import RedisSettings
from src.domain.agents.fusionsearch.schemas import AskRequest, AskResponse
from src.domain.cache.keys import build_exact_cache_key
from src.domain.cache.scoring import CacheConfidenceBreakdown
from src.domain.cache.semantic import SemanticCacheClient
from src.domain.jinaai.jina_client import JinaEmbeddingsClient

logger = logging.getLogger(__name__)


@dataclass
class CacheLookupResult:
    """Result of a layered cache lookup."""

    response: Optional[AskResponse] = None
    query_embedding: Optional[list[float]] = None
    hit_type: Optional[Literal["exact", "confidence"]] = None
    confidence: Optional[CacheConfidenceBreakdown] = None


class ExactCacheClient:
    """Redis-based exact match cache for RAG queries."""

    def __init__(self, redis_client: redis.Redis, settings: RedisSettings):
        self.redis = redis_client
        self.settings = settings
        self.ttl = timedelta(hours=settings.ttl_hours)

    async def find_cached_response(self, request: AskRequest) -> Optional[AskResponse]:
        """Find cached response for exact query match."""
        try:
            cache_key = build_exact_cache_key(request)
            cached_response = self.redis.get(cache_key)

            if cached_response:
                try:
                    response_data = json.loads(cached_response)
                    logger.info("Cache hit for exact query match")
                    return AskResponse(**response_data)
                except json.JSONDecodeError as e:
                    logger.warning("Failed to deserialize cached response: %s", e)
                    return None

            return None

        except Exception as e:
            logger.error("Error checking exact cache: %s", e)
            return None

    async def store_response(self, request: AskRequest, response: AskResponse) -> bool:
        """Store response for exact query matching."""
        try:
            cache_key = build_exact_cache_key(request)
            success = self.redis.set(cache_key, response.model_dump_json(), ex=self.ttl)

            if success:
                logger.info("Stored response in exact cache with key %s...", cache_key[:16])
                return True

            logger.warning("Failed to store response in exact cache")
            return False

        except Exception as e:
            logger.error("Error storing in exact cache: %s", e)
            return False


class CacheClient:
    """Layered RAG cache: exact match first, then confidence-based fuzzy + semantic matching."""

    def __init__(
        self,
        exact_cache: ExactCacheClient,
        semantic_cache: Optional[SemanticCacheClient],
        embeddings_client: Optional[JinaEmbeddingsClient],
        settings: RedisSettings,
    ):
        self.exact = exact_cache
        self.semantic = semantic_cache
        self.embeddings = embeddings_client
        self.settings = settings

    async def _embed_query(self, request: AskRequest) -> Optional[list[float]]:
        if not self.embeddings:
            return None

        try:
            return await self.embeddings.embed_query(request.query)
        except Exception as e:
            logger.warning("Failed to generate query embedding for semantic cache: %s", e)
            return None

    async def lookup(self, request: AskRequest) -> CacheLookupResult:
        """Check exact cache, then semantic cache. Returns embedding for reuse."""
        exact_hit = await self.exact.find_cached_response(request)
        if exact_hit:
            return CacheLookupResult(response=exact_hit, hit_type="exact")

        if not self.settings.semantic_cache_enabled or not self.semantic or not self.semantic.is_ready:
            return CacheLookupResult()

        query_embedding = await self._embed_query(request)
        if not query_embedding:
            return CacheLookupResult()

        semantic_hit, confidence_breakdown = await self.semantic.find_cached_response(request, query_embedding)
        if semantic_hit:
            return CacheLookupResult(
                response=semantic_hit,
                query_embedding=query_embedding,
                hit_type="confidence",
                confidence=confidence_breakdown,
            )

        return CacheLookupResult(query_embedding=query_embedding)

    async def store(
        self,
        request: AskRequest,
        response: AskResponse,
        query_embedding: Optional[list[float]] = None,
    ) -> None:
        """Store response in exact and semantic caches."""
        await self.exact.store_response(request, response)

        if not self.settings.semantic_cache_enabled or not self.semantic or not self.semantic.is_ready:
            return

        embedding = query_embedding or await self._embed_query(request)
        if embedding:
            await self.semantic.store_response(request, response, embedding)

    async def find_cached_response(self, request: AskRequest) -> Optional[AskResponse]:
        """Backward-compatible cache lookup."""
        result = await self.lookup(request)
        return result.response

    async def store_response(
        self,
        request: AskRequest,
        response: AskResponse,
        query_embedding: Optional[list[float]] = None,
    ) -> bool:
        """Backward-compatible cache store."""
        await self.store(request, response, query_embedding=query_embedding)
        return True
