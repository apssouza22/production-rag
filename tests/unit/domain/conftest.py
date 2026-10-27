"""Shared pytest fixtures for retrieval tests (agents and rerank)."""

from unittest.mock import AsyncMock, Mock

import pytest
from src.domain.jinaai.jina import JinaRerankResult


@pytest.fixture
def mock_opensearch_client():
    """Mock OpenSearch client with sample search results."""
    client = Mock()
    client.search_unified = Mock(
        return_value={
            "hits": [
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
            ],
            "total": 2,
        }
    )
    return client


@pytest.fixture
def mock_jina_embeddings_client():
    """Mock Jina embeddings client."""
    client = AsyncMock()
    client.embed_query = AsyncMock(return_value=[0.1] * 1024)
    return client


@pytest.fixture
def mock_jina_reranker_client():
    """Mock Jina reranker client."""
    client = AsyncMock()
    client.is_configured = True
    client.rerank = AsyncMock(
        return_value=[
            JinaRerankResult(index=0, relevance_score=0.98),
            JinaRerankResult(index=1, relevance_score=0.72),
        ]
    )
    return client
