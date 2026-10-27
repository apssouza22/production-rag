"""Pytest fixtures for agent tests.

The retrieval mocks (OpenSearch, Jina embeddings and reranker) live in ``tests/unit/domain/conftest.py``,
shared with the rerank tests.
"""

import pytest
from src.domain.rerank.config import RerankSearchConfig
from src.domain.rerank.factory import make_rerank_search_service


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
        config=RerankSearchConfig(use_hybrid=True, rerank_enabled=False),
    )


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
