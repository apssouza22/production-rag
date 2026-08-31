from unittest.mock import AsyncMock, Mock

import pytest

from src.config import RedisSettings
from src.domain.cache.client import CacheClient, ExactCacheClient
from src.domain.cache.keys import build_exact_cache_key, build_params_hash
from src.domain.cache.scoring import CacheConfidenceBreakdown
from src.domain.cache.semantic import SemanticCacheClient, embedding_to_bytes
from src.domain.opensearch.schemas import HybridSearchRequest, SearchHit, SearchResponse


@pytest.fixture
def search_request() -> HybridSearchRequest:
    return HybridSearchRequest(
        query="What are transformers?",
        size=3,
        from_=0,
        use_hybrid=True,
        categories=["cs.AI"],
        latest_papers=False,
        min_score=0.0,
    )


@pytest.fixture
def search_response() -> SearchResponse:
    return SearchResponse(
        query="What are transformers?",
        total=1,
        hits=[
            SearchHit(
                arxiv_id="1706.03762",
                title="Attention Is All You Need",
                authors="Vaswani et al.",
                abstract="Transformer architecture",
                published_date="2017-06-12",
                pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
                score=0.95,
            )
        ],
        size=3,
        **{"from": 0},
        search_mode="hybrid",
    )


@pytest.fixture
def confidence_breakdown() -> CacheConfidenceBreakdown:
    return CacheConfidenceBreakdown(
        confidence=0.78,
        exact_score=0.0,
        fuzzy_score=0.72,
        semantic_score=0.85,
        matched_query="What are transformer models?",
    )


class TestCacheKeys:
    def test_params_hash_ignores_query_text(self, search_request: HybridSearchRequest):
        other_request = search_request.model_copy(update={"query": "Explain transformer architecture"})
        assert build_params_hash(search_request) == build_params_hash(other_request)

    def test_exact_cache_key_changes_with_query(self, search_request: HybridSearchRequest):
        other_request = search_request.model_copy(update={"query": "Explain transformer architecture"})
        assert build_exact_cache_key(search_request) != build_exact_cache_key(other_request)

    def test_exact_cache_key_changes_with_params(self, search_request: HybridSearchRequest):
        other_request = search_request.model_copy(update={"size": 5})
        assert build_exact_cache_key(search_request) != build_exact_cache_key(other_request)


class TestEmbeddingBytes:
    def test_round_trip(self):
        vector = [0.1, 0.2, 0.3]
        packed = embedding_to_bytes(vector)
        assert len(packed) == len(vector) * 4


class TestCacheClient:
    @pytest.mark.asyncio
    async def test_lookup_returns_exact_hit_without_embedding(self, search_request, search_response):
        exact_cache = Mock(spec=ExactCacheClient)
        exact_cache.find_cached_response = AsyncMock(return_value=search_response)

        cache_client = CacheClient(
            exact_cache=exact_cache,
            semantic_cache=None,
            embeddings_client=None,
            settings=RedisSettings(),
        )

        result = await cache_client.lookup(search_request)

        assert result.response == search_response
        assert result.hit_type == "exact"
        assert result.query_embedding is None

    @pytest.mark.asyncio
    async def test_lookup_uses_confidence_cache_after_exact_miss(
        self, search_request, search_response, confidence_breakdown
    ):
        exact_cache = Mock(spec=ExactCacheClient)
        exact_cache.find_cached_response = AsyncMock(return_value=None)

        semantic_cache = Mock(spec=SemanticCacheClient)
        semantic_cache.is_ready = True
        semantic_cache.find_cached_response = AsyncMock(return_value=(search_response, confidence_breakdown))

        embeddings_client = Mock()
        embeddings_client.embed_query = AsyncMock(return_value=[0.1] * 1024)

        cache_client = CacheClient(
            exact_cache=exact_cache,
            semantic_cache=semantic_cache,
            embeddings_client=embeddings_client,
            settings=RedisSettings(semantic_cache_enabled=True),
        )

        result = await cache_client.lookup(search_request)

        assert result.response == search_response
        assert result.hit_type == "confidence"
        assert result.confidence == confidence_breakdown
        assert result.query_embedding == [0.1] * 1024
        embeddings_client.embed_query.assert_awaited_once_with(search_request.query)

    @pytest.mark.asyncio
    async def test_lookup_returns_embedding_on_confidence_miss(self, search_request):
        exact_cache = Mock(spec=ExactCacheClient)
        exact_cache.find_cached_response = AsyncMock(return_value=None)

        semantic_cache = Mock(spec=SemanticCacheClient)
        semantic_cache.is_ready = True
        semantic_cache.find_cached_response = AsyncMock(return_value=(None, None))

        embeddings_client = Mock()
        embeddings_client.embed_query = AsyncMock(return_value=[0.2] * 1024)

        cache_client = CacheClient(
            exact_cache=exact_cache,
            semantic_cache=semantic_cache,
            embeddings_client=embeddings_client,
            settings=RedisSettings(semantic_cache_enabled=True),
        )

        result = await cache_client.lookup(search_request)

        assert result.response is None
        assert result.hit_type is None
        assert result.query_embedding == [0.2] * 1024

    @pytest.mark.asyncio
    async def test_store_writes_to_both_layers(self, search_request, search_response):
        exact_cache = Mock(spec=ExactCacheClient)
        exact_cache.store_response = AsyncMock(return_value=True)

        semantic_cache = Mock(spec=SemanticCacheClient)
        semantic_cache.is_ready = True
        semantic_cache.store_response = AsyncMock(return_value=True)

        embedding = [0.3] * 1024
        cache_client = CacheClient(
            exact_cache=exact_cache,
            semantic_cache=semantic_cache,
            embeddings_client=None,
            settings=RedisSettings(semantic_cache_enabled=True),
        )

        await cache_client.store(search_request, search_response, query_embedding=embedding)

        exact_cache.store_response.assert_awaited_once_with(search_request, search_response)
        semantic_cache.store_response.assert_awaited_once_with(search_request, search_response, embedding)
