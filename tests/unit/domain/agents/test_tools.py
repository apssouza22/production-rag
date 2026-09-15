from unittest.mock import AsyncMock, Mock

import pytest
from src.agents.fusionsearch.config import GraphConfig
from src.agents.fusionsearch.tools import create_retriever_tool


@pytest.fixture
def graph_config():
    return GraphConfig(top_k=2, use_hybrid=True, rerank_enabled=False)


@pytest.mark.asyncio
async def test_create_retriever_tool_basic(
    mock_opensearch_client,
    mock_jina_embeddings_client,
    mock_rerank_search_service,
    graph_config,
):
    """Test basic retriever tool creation and invocation."""
    tool = create_retriever_tool(
        rerank_search_service=mock_rerank_search_service,
        graph_config=graph_config,
    )

    assert tool.name == "retrieve_papers"
    assert "Search and return relevant arXiv research papers" in tool.description

    result = await tool.ainvoke({"query": "machine learning"})

    assert isinstance(result, dict)
    assert "formatted_context" in result
    assert "retrieval_confidence" in result
    assert result["document_count"] == 2
    assert len(result["sources"]) == 2
    assert "1706.03762" in result["formatted_context"]

    mock_jina_embeddings_client.embed_query.assert_called_once_with("machine learning")

    mock_opensearch_client.search_unified.assert_called_once()
    call_args = mock_opensearch_client.search_unified.call_args
    assert call_args.kwargs["query"] == "machine learning"
    assert call_args.kwargs["size"] == 2
    assert call_args.kwargs["use_hybrid"] is True


@pytest.mark.asyncio
async def test_retriever_tool_empty_results(
    mock_opensearch_client,
    mock_rerank_search_service,
    graph_config,
):
    """Test retriever tool with no results."""
    mock_opensearch_client.search_unified = Mock(return_value={"hits": []})

    tool = create_retriever_tool(
        rerank_search_service=mock_rerank_search_service,
        graph_config=graph_config,
    )

    result = await tool.ainvoke({"query": "nonexistent topic"})

    assert isinstance(result, dict)
    assert result["document_count"] == 0
    assert result["retrieval_confidence"] == 0.0
    assert result["formatted_context"] == ""


@pytest.mark.asyncio
async def test_retriever_tool_custom_top_k(
    mock_opensearch_client,
    mock_rerank_search_service,
):
    """Test retriever tool with custom top_k parameter."""
    graph_config = GraphConfig(top_k=5, use_hybrid=False, rerank_enabled=False)
    mock_rerank_search_service.config.use_hybrid = False
    mock_rerank_search_service.config.rerank_enabled = False

    tool = create_retriever_tool(
        rerank_search_service=mock_rerank_search_service,
        graph_config=graph_config,
    )

    await tool.ainvoke({"query": "test query"})

    call_args = mock_opensearch_client.search_unified.call_args
    assert call_args.kwargs["size"] == 5
    assert call_args.kwargs["use_hybrid"] is False


@pytest.mark.asyncio
async def test_retriever_tool_metadata_fields(
    mock_opensearch_client,
    mock_rerank_search_service,
    graph_config,
):
    """Test that structured source metadata is present."""
    mock_opensearch_client.search_unified = Mock(
        return_value={
            "hits": [
                {
                    "chunk_text": "Test content",
                    "arxiv_id": "2301.00001",
                    "title": "Test Paper",
                    "authors": "Author One, Author Two",
                    "score": 0.95,
                    "section_name": "Introduction",
                }
            ]
        }
    )

    tool = create_retriever_tool(
        rerank_search_service=mock_rerank_search_service,
        graph_config=graph_config,
    )

    result = await tool.ainvoke({"query": "test"})

    source = result["sources"][0]
    assert source["arxiv_id"] == "2301.00001"
    assert source["title"] == "Test Paper"
    assert source["url"] == "https://arxiv.org/pdf/2301.00001.pdf"


@pytest.mark.asyncio
async def test_retriever_tool_reranks_candidates(
    mock_opensearch_client,
    mock_jina_reranker_client,
    mock_rerank_search_service,
):
    """Test retriever tool reranks a larger candidate pool."""
    graph_config = GraphConfig(top_k=2, use_hybrid=True, rerank_enabled=True)
    mock_rerank_search_service.config.rerank_enabled = True

    tool = create_retriever_tool(
        rerank_search_service=mock_rerank_search_service,
        graph_config=graph_config,
    )

    result = await tool.ainvoke({"query": "machine learning"})

    mock_opensearch_client.search_unified.assert_called_once()
    assert mock_opensearch_client.search_unified.call_args.kwargs["size"] == 4

    mock_jina_reranker_client.rerank.assert_called_once()
    rerank_kwargs = mock_jina_reranker_client.rerank.call_args.kwargs
    assert rerank_kwargs["query"] == "machine learning"
    assert rerank_kwargs["top_n"] == 2
    assert len(rerank_kwargs["documents"]) == 2

    assert result["document_count"] == 2
    assert result["rerank_applied"] is True


@pytest.mark.asyncio
async def test_retriever_tool_rerank_failure_falls_back(
    mock_jina_reranker_client,
    mock_rerank_search_service,
):
    """Test retriever tool falls back to search order when reranking fails."""
    graph_config = GraphConfig(top_k=2, rerank_enabled=True)
    mock_rerank_search_service.config.rerank_enabled = True
    mock_jina_reranker_client.rerank = AsyncMock(side_effect=Exception("rerank failed"))

    tool = create_retriever_tool(
        rerank_search_service=mock_rerank_search_service,
        graph_config=graph_config,
    )

    result = await tool.ainvoke({"query": "machine learning"})

    assert result["document_count"] == 2
    assert result["rerank_applied"] is False
