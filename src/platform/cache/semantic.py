import json
import logging
import struct
import uuid
from datetime import timedelta
from typing import List, Optional

import redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from src.config import RedisSettings
from src.platform.cache.keys import build_params_hash
from src.platform.cache.scoring import CacheConfidenceBreakdown, compute_confidence
from src.domain.opensearch.schemas import HybridSearchRequest, SearchResponse

logger = logging.getLogger(__name__)

SEMANTIC_CACHE_INDEX = "hybrid_search_semantic_cache_idx"
SEMANTIC_CACHE_PREFIX = "hybrid_search_semantic_cache:"


def embedding_to_bytes(embedding: List[float]) -> bytes:
    """Pack a float embedding vector into Redis vector field bytes."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _decode_field(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


class SemanticCacheClient:
    """Redis Stack vector cache with confidence scoring (exact + fuzzy + semantic)."""

    def __init__(self, redis_client: redis.Redis, settings: RedisSettings):
        self.redis = redis_client
        self.settings = settings
        self.ttl = timedelta(hours=settings.ttl_hours)
        self.candidate_min_similarity = settings.semantic_similarity_threshold
        self.confidence_threshold = settings.confidence_threshold
        self.search_top_k = settings.semantic_search_top_k
        self.embedding_dimensions = settings.embedding_dimensions
        self._index_ready = False
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Create the RediSearch vector index if it does not exist."""
        try:
            self.redis.ft(SEMANTIC_CACHE_INDEX).info()
            self._index_ready = True
            logger.info("Hybrid search semantic cache index already exists")
            return
        except ResponseError:
            pass

        schema = (
            TagField("params_hash"),
            TextField("query"),
            TextField("response"),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.embedding_dimensions,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        )
        definition = IndexDefinition(prefix=[SEMANTIC_CACHE_PREFIX], index_type=IndexType.HASH)

        try:
            self.redis.ft(SEMANTIC_CACHE_INDEX).create_index(schema, definition=definition)
            self._index_ready = True
            logger.info("Hybrid search semantic cache index created")
        except ResponseError as e:
            if "Index already exists" in str(e):
                self._index_ready = True
                logger.info("Hybrid search semantic cache index already exists")
                return
            raise

    @property
    def is_ready(self) -> bool:
        return self._index_ready

    def _score_candidate(
        self,
        query: str,
        cached_query: str,
        semantic_similarity: float,
    ) -> CacheConfidenceBreakdown:
        return compute_confidence(
            query,
            cached_query,
            semantic_similarity,
            weight_exact=self.settings.cache_weight_exact,
            weight_fuzzy=self.settings.cache_weight_fuzzy,
            weight_semantic=self.settings.cache_weight_semantic,
        )

    def _select_best_candidate(
        self,
        query: str,
        candidates: list[tuple[str, str, float]],
    ) -> Optional[tuple[CacheConfidenceBreakdown, str]]:
        best_match: Optional[tuple[CacheConfidenceBreakdown, str]] = None

        for cached_query, response_json, semantic_similarity in candidates:
            if semantic_similarity < self.candidate_min_similarity:
                continue

            breakdown = self._score_candidate(query, cached_query, semantic_similarity)
            logger.debug(
                "Cache candidate %r → exact=%.2f fuzzy=%.2f semantic=%.2f confidence=%.3f",
                cached_query[:80],
                breakdown.exact_score,
                breakdown.fuzzy_score,
                breakdown.semantic_score,
                breakdown.confidence,
            )

            if best_match is None or breakdown.confidence > best_match[0].confidence:
                best_match = (breakdown, response_json)

        if best_match and best_match[0].confidence >= self.confidence_threshold:
            return best_match

        return None

    async def find_cached_response(
        self,
        request: HybridSearchRequest,
        query_embedding: List[float],
    ) -> tuple[Optional[SearchResponse], Optional[CacheConfidenceBreakdown]]:
        """Find a cached response using confidence-based fuzzy + semantic scoring."""
        if not self._index_ready:
            return None, None

        if len(query_embedding) != self.embedding_dimensions:
            logger.warning(
                "Query embedding dimension %s does not match cache dimension %s",
                len(query_embedding),
                self.embedding_dimensions,
            )
            return None, None

        try:
            params_hash = build_params_hash(request)
            vec_bytes = embedding_to_bytes(query_embedding)

            query = (
                Query(
                    f"(@params_hash:{{{params_hash}}})=>[KNN {self.search_top_k} @embedding $vec AS distance]"
                )
                .sort_by("distance")
                .return_fields("response", "distance", "query")
                .dialect(2)
            )

            results = self.redis.ft(SEMANTIC_CACHE_INDEX).search(query, query_params={"vec": vec_bytes})
            if not results.docs:
                return None, None

            candidates: list[tuple[str, str, float]] = []
            for doc in results.docs:
                distance = float(getattr(doc, "distance", 2.0))
                semantic_similarity = max(0.0, 1.0 - distance)
                cached_query = _decode_field(doc.query)
                response_json = _decode_field(doc.response)
                candidates.append((cached_query, response_json, semantic_similarity))

            best_match = self._select_best_candidate(request.query, candidates)
            if not best_match:
                logger.debug("Semantic cache miss: no candidate met confidence threshold")
                return None, None

            best_breakdown, matched_response = best_match

            logger.info(
                "Confidence cache hit (confidence=%.3f exact=%.2f fuzzy=%.2f semantic=%.2f matched_query=%r)",
                best_breakdown.confidence,
                best_breakdown.exact_score,
                best_breakdown.fuzzy_score,
                best_breakdown.semantic_score,
                best_breakdown.matched_query[:80],
            )
            return SearchResponse(**json.loads(matched_response)), best_breakdown

        except Exception as e:
            logger.error("Error checking semantic cache: %s", e)
            return None, None

    async def store_response(
        self,
        request: HybridSearchRequest,
        response: SearchResponse,
        query_embedding: List[float],
    ) -> bool:
        """Store a response for semantic similarity lookups."""
        if not self._index_ready:
            return False

        if len(query_embedding) != self.embedding_dimensions:
            logger.warning("Skipping semantic cache store due to embedding dimension mismatch")
            return False

        try:
            cache_key = f"{SEMANTIC_CACHE_PREFIX}{uuid.uuid4().hex}"
            mapping = {
                "params_hash": build_params_hash(request),
                "query": request.query,
                "response": response.model_dump_json(by_alias=True),
                "embedding": embedding_to_bytes(query_embedding),
            }

            pipe = self.redis.pipeline()
            pipe.hset(cache_key, mapping=mapping)
            pipe.expire(cache_key, int(self.ttl.total_seconds()))
            pipe.execute()

            logger.info("Stored response in semantic cache with key %s...", cache_key[:24])
            return True

        except Exception as e:
            logger.error("Error storing in semantic cache: %s", e)
            return False
