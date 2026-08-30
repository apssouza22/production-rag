import pytest

from src.config import RedisSettings
from src.domain.cache.semantic import SemanticCacheClient


@pytest.fixture
def settings() -> RedisSettings:
    return RedisSettings(
        semantic_similarity_threshold=0.7,
        confidence_threshold=0.90,
        cache_weight_exact=0.10,
        cache_weight_fuzzy=0.20,
        cache_weight_semantic=0.70,
    )


@pytest.fixture
def semantic_client(settings: RedisSettings) -> SemanticCacheClient:
    client = SemanticCacheClient.__new__(SemanticCacheClient)
    client.settings = settings
    client.candidate_min_similarity = settings.semantic_similarity_threshold
    client.confidence_threshold = settings.confidence_threshold
    client.search_top_k = settings.semantic_search_top_k
    client.embedding_dimensions = settings.embedding_dimensions
    client._index_ready = True
    return client


class TestSemanticCandidateSelection:
    def test_selects_highest_confidence_candidate(self, semantic_client: SemanticCacheClient):
        candidates = [
            ("What are transformers?", "{}", 0.98),
            ("Explain transformer architecture in ML", "{}", 0.80),
            ("How does backpropagation work?", "{}", 0.75),
        ]

        best_match = semantic_client._select_best_candidate(
            "What are transformers?",
            candidates,
        )

        assert best_match is not None
        breakdown, _ = best_match
        assert breakdown.matched_query == "What are transformers?"
        assert breakdown.confidence >= 0.90

    def test_rejects_candidates_below_confidence_threshold(self, semantic_client: SemanticCacheClient):
        candidates = [
            ("Completely unrelated topic about databases", "{}", 0.71),
        ]

        best_match = semantic_client._select_best_candidate(
            "What are transformers?",
            candidates,
        )

        assert best_match is None

    def test_rejects_borderline_match_below_confidence_threshold(self, semantic_client: SemanticCacheClient):
        candidates = [
            ("What are transformers in machine learning?", "{}", 0.72),
        ]

        best_match = semantic_client._select_best_candidate(
            "What are transformers in ML?",
            candidates,
        )

        assert best_match is None
