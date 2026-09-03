import pytest
from unittest.mock import AsyncMock, Mock
from langchain_core.documents import Document

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

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(doc, Document) for doc in result)

    first_doc = result[0]
    assert first_doc.page_content == "Transformers are neural network architectures based on self-attention mechanisms."
    assert first_doc.metadata["arxiv_id"] == "1706.03762"
    assert first_doc.metadata["title"] == "Attention Is All You Need"
    assert first_doc.metadata["score"] == 0.95

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

    assert isinstance(result, list)
    assert len(result) == 0


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
    """Test that all expected metadata fields are present."""
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

    doc = result[0]
    assert "arxiv_id" in doc.metadata
    assert "title" in doc.metadata
    assert "authors" in doc.metadata
    assert "score" in doc.metadata
    assert "source" in doc.metadata
    assert "section" in doc.metadata


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

    assert len(result) == 2
    assert result[0].metadata["reranked"] is True
    assert result[0].metadata["rerank_score"] == 0.98


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

    assert len(result) == 2
    assert result[0].metadata["reranked"] is False
