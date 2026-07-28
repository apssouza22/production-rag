"""Tests for Jina reranker client."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from src.domain.jinaai.jina_reranker_client import JinaRerankerClient


@pytest.fixture
def reranker_client():
    return JinaRerankerClient(api_key="test-api-key")


@pytest.mark.asyncio
async def test_rerank_returns_ordered_results(reranker_client):
    """Test rerank parses API response and returns results."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "model": "jina-reranker-v2-base-multilingual",
        "results": [
            {"index": 1, "relevance_score": 0.91},
            {"index": 0, "relevance_score": 0.42},
        ],
        "usage": {"total_tokens": 100},
    }
    mock_response.raise_for_status = Mock()

    with patch.object(reranker_client.client, "post", new_callable=AsyncMock, return_value=mock_response):
        results = await reranker_client.rerank(
            query="attention mechanisms",
            documents=["doc one", "doc two"],
            top_n=2,
        )

    assert len(results) == 2
    assert results[0].index == 1
    assert results[0].relevance_score == 0.91


@pytest.mark.asyncio
async def test_rerank_empty_documents(reranker_client):
    """Test rerank with empty document list."""
    results = await reranker_client.rerank(query="test", documents=[], top_n=3)
    assert results == []


@pytest.mark.asyncio
async def test_rerank_missing_api_key_raises():
    """Test rerank without API key raises ValueError."""
    client = JinaRerankerClient(api_key="")

    with pytest.raises(ValueError, match="Jina API key is not configured"):
        await client.rerank(query="test", documents=["doc"], top_n=1)


@pytest.mark.asyncio
async def test_rerank_accepts_score_alias(reranker_client):
    """Test rerank accepts Jina API score field alias."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "model": "jina-reranker-v2-base-multilingual",
        "results": [{"index": 0, "score": 0.75}],
    }
    mock_response.raise_for_status = Mock()

    with patch.object(reranker_client.client, "post", new_callable=AsyncMock, return_value=mock_response):
        results = await reranker_client.rerank(query="test", documents=["doc"], top_n=1)

    assert results[0].relevance_score == 0.75


@pytest.mark.asyncio
async def test_rerank_http_error_propagates(reranker_client):
    """Test HTTP errors from Jina API are propagated."""
    with patch.object(
        reranker_client.client,
        "post",
        new_callable=AsyncMock,
        side_effect=httpx.HTTPError("API error"),
    ):
        with pytest.raises(httpx.HTTPError):
            await reranker_client.rerank(
                query="test",
                documents=["doc one"],
                top_n=1,
            )
