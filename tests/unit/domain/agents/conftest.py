"""Shared pytest fixtures for agent and retrieval tests."""

from unittest.mock import AsyncMock, Mock

import pytest

from src.agents.fusionsearch.retrieval_settings import RetrievalSettings
from src.domain.jinaai.jina import JinaRerankResult
from src.domain.rerank.factory import make_rerank_search_service


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


@pytest.fixture
def mock_rerank_search_service(
    mock_opensearch_client,
    mock_jina_embeddings_client,
    mock_jina_reranker_client,
):
    """Mock rerank search service wired from retrieval dependencies."""
    return make_rerank_search_service(
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        reranker_client=mock_jina_reranker_client,
    )


@pytest.fixture
def mock_ollama_client():
    """Mock LLM client."""
    client = Mock()
    client.get_langchain_model = Mock(return_value=AsyncMock())
    return client


@pytest.fixture
def retrieval_settings():
    """Default retrieval settings for tool tests."""
    return RetrievalSettings(top_k=2, use_hybrid=True, rerank_enabled=False)


@pytest.fixture
def sample_human_message():
    """Sample human message for node tests."""
    from langchain_core.messages import HumanMessage

    return HumanMessage(content="What are attention mechanisms in transformers?")


@pytest.fixture
def sample_ai_message():
    """Sample AI message for node tests."""
    from langchain_core.messages import AIMessage

    return AIMessage(content="What is machine learning?")


@pytest.fixture
def sample_tool_message():
    """Sample tool message for node tests."""
    from langchain_core.messages import ToolMessage

    return ToolMessage(
        content="Transformers are neural network architectures based on self-attention.",
        tool_call_id="retrieve_1",
        name="retrieve_papers",
    )
