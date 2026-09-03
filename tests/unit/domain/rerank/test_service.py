"""Tests for rerank search service."""

from unittest.mock import AsyncMock, Mock

import pytest

from src.domain.jinaai.jina import JinaRerankResult
from src.domain.rerank.config import RerankSearchConfig
from src.domain.rerank.service import RERANK_CANDIDATE_MULTIPLIER, RerankSearchService


@pytest.fixture
def sample_hits():
    return [
        {
            "chunk_text": "Transformers are neural network architectures based on self-attention mechanisms.",
            "arxiv_id": "1706.03762",
            "title": "Attention Is All You Need",
            "authors": "Vaswani et al.",
            "score": 0.95,
            "section_name": "Introduction",
        },
        {
            "chunk_text": "BERT is a bidirectional transformer pre-trained on large corpora.",
            "arxiv_id": "1810.04805",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": "Devlin et al.",
            "score": 0.88,
            "section_name": "Abstract",
        },
    ]


@pytest.fixture
def rerank_search_service(mock_opensearch_client, mock_jina_embeddings_client, mock_jina_reranker_client):
    return RerankSearchService(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        reranker_client=mock_jina_reranker_client,
        config=RerankSearchConfig(use_hybrid=True, rerank_enabled=False),
    )


@pytest.mark.asyncio
async def test_search_without_rerank_returns_before_and_after(rerank_search_service, sample_hits):
    """Test search returns identical before/after lists when reranking is disabled."""
    result = await rerank_search_service.search(query="machine learning", top_k=2)

    assert result.query == "machine learning"
    assert result.search_mode == "hybrid"
    assert result.rerank_applied is False
    assert len(result.before_rerank) == 2
    assert len(result.after_rerank) == 2
    assert result.after_rerank[0].arxiv_id == sample_hits[0]["arxiv_id"]
    assert result.before_rerank[0].rank == 0
    assert result.after_rerank[0].rank == 0


@pytest.mark.asyncio
async def test_search_with_rerank_returns_reordered_after_list(
    mock_opensearch_client,
    mock_jina_embeddings_client,
    mock_jina_reranker_client,
):
    """Test reranking preserves before list and returns reranked after list."""
    service = RerankSearchService(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        reranker_client=mock_jina_reranker_client,
        config=RerankSearchConfig(rerank_enabled=True),
    )
    mock_opensearch_client.search_unified = Mock(
        return_value={
            "hits": [
                {
                    "chunk_text": "doc-a",
                    "arxiv_id": "1111.11111",
                    "title": "Paper A",
                    "score": 0.9,
                },
                {
                    "chunk_text": "doc-b",
                    "arxiv_id": "2222.22222",
                    "title": "Paper B",
                    "score": 0.8,
                },
                {
                    "chunk_text": "doc-c",
                    "arxiv_id": "3333.33333",
                    "title": "Paper C",
                    "score": 0.7,
                },
            ],
            "total": 3,
        }
    )
    mock_jina_reranker_client.rerank = AsyncMock(
        return_value=[
            JinaRerankResult(index=2, relevance_score=0.99),
            JinaRerankResult(index=0, relevance_score=0.75),
        ]
    )

    result = await service.search(query="test query", top_k=2)

    mock_opensearch_client.search_unified.assert_called_once()
    assert mock_opensearch_client.search_unified.call_args.kwargs["size"] == 2 * RERANK_CANDIDATE_MULTIPLIER
    mock_jina_reranker_client.rerank.assert_called_once()

    assert result.rerank_applied is True
    assert len(result.before_rerank) == 3
    assert result.before_rerank[0].arxiv_id == "1111.11111"
    assert len(result.after_rerank) == 2
    assert result.after_rerank[0].arxiv_id == "3333.33333"
    assert result.after_rerank[0].rerank_score == 0.99
    assert result.after_rerank[0].original_rank == 2


@pytest.mark.asyncio
async def test_search_rerank_failure_falls_back_to_search_order(
    mock_opensearch_client,
    mock_jina_embeddings_client,
    mock_jina_reranker_client,
):
    """Test rerank failure keeps original order in after list."""
    service = RerankSearchService(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        reranker_client=mock_jina_reranker_client,
        config=RerankSearchConfig(rerank_enabled=True),
    )
    mock_jina_reranker_client.rerank = AsyncMock(side_effect=Exception("rerank failed"))

    result = await service.search(query="machine learning", top_k=2)

    assert result.rerank_applied is False
    assert len(result.before_rerank) == 2
    assert len(result.after_rerank) == 2
    assert result.after_rerank[0].arxiv_id == result.before_rerank[0].arxiv_id


@pytest.mark.asyncio
async def test_search_bm25_mode(mock_opensearch_client, mock_jina_embeddings_client, mock_jina_reranker_client):
    """Test BM25 search mode is reflected in the result."""
    service = RerankSearchService(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        reranker_client=mock_jina_reranker_client,
        config=RerankSearchConfig(use_hybrid=False, rerank_enabled=False),
    )

    await service.search(query="test", top_k=2)

    call_args = mock_opensearch_client.search_unified.call_args
    assert call_args.kwargs["use_hybrid"] is False

    result = await service.search(query="test", top_k=2)
    assert result.search_mode == "bm25"
