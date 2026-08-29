"""Tests for AgenticRAGGraph node methods."""

import pytest
from unittest.mock import AsyncMock, Mock
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.fusionsearch.config import GraphConfig
from src.agents.fusionsearch.graph import AgenticRAGGraph
from src.agents.fusionsearch.utils import get_latest_query, get_latest_context
from src.agents.fusionsearch.models import GuardrailScoring, GradeDocuments
from src.agents.fusionsearch.state import AgentState


@pytest.fixture
def graph(mock_ollama_client, mock_opensearch_client, mock_jina_embeddings_client):
    """AgenticRAGGraph with mocked dependencies."""
    from src.agents.fusionsearch.retrieval_settings import RetrievalSettings

    config = GraphConfig(
        model="gpt-4o-mini",
        temperature=0.0,
        max_retrieval_attempts=2,
        guardrail_threshold=60,
    )
    return AgenticRAGGraph(
        llm_client=mock_ollama_client,
        opensearch_client=mock_opensearch_client,
        embeddings_client=mock_jina_embeddings_client,
        retrieval_settings=RetrievalSettings(),
        config=config,
    )


class TestGuardrailNode:
    def test_continue_after_guardrail_pass(self, graph):
        state: AgentState = {
            "messages": [],
            "retrieval_attempts": 0,
            "guardrail_result": GuardrailScoring(score=75, reason="Pass"),
        }

        assert graph.continue_after_guardrail(state) == "continue"

    def test_continue_after_guardrail_fail(self, graph):
        state: AgentState = {
            "messages": [],
            "retrieval_attempts": 0,
            "guardrail_result": GuardrailScoring(score=30, reason="Fail"),
        }

        assert graph.continue_after_guardrail(state) == "out_of_scope"


class TestRetrieveNode:
    @pytest.mark.asyncio
    async def test_retrieve_creates_tool_call(self, graph, sample_human_message):
        state: AgentState = {
            "messages": [sample_human_message],
            "retrieval_attempts": 0,
        }

        result = await graph.retrieve(state)

        assert result["retrieval_attempts"] == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].tool_calls[0]["name"] == "retrieve_papers"

    @pytest.mark.asyncio
    async def test_retrieve_max_attempts_reached(self, graph, sample_human_message):
        state: AgentState = {
            "messages": [sample_human_message],
            "retrieval_attempts": 2,
        }

        result = await graph.retrieve(state)

        content_lower = result["messages"][0].content.lower()
        assert "apologize" in content_lower or "couldn't find" in content_lower


class TestGradeDocumentsNode:
    @pytest.mark.asyncio
    async def test_grade_documents_relevant(self, graph, sample_human_message, sample_tool_message):
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(
            return_value=GradeDocuments(
                binary_score="yes",
                reasoning="Document discusses transformers which is relevant",
            )
        )
        graph.llm_client.get_langchain_model = Mock(return_value=mock_llm)

        state: AgentState = {
            "messages": [sample_human_message, sample_tool_message],
            "retrieval_attempts": 1,
        }

        result = await graph.grade_documents(state)

        assert "grading_results" in result

    @pytest.mark.asyncio
    async def test_grade_documents_not_relevant(self, graph, sample_human_message, sample_tool_message):
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(
            return_value=GradeDocuments(
                binary_score="no",
                reasoning="Document is not relevant to the query",
            )
        )
        graph.llm_client.get_langchain_model = Mock(return_value=mock_llm)

        state: AgentState = {
            "messages": [sample_human_message, sample_tool_message],
            "retrieval_attempts": 1,
        }

        result = await graph.grade_documents(state)

        assert "grading_results" in result


class TestRewriteQueryNode:
    @pytest.mark.asyncio
    async def test_rewrite_query_success(self, graph, sample_human_message):
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(
            return_value=Mock(
                rewritten_query="What are transformer neural network architectures?",
                reasoning="Expanded technical terms",
            )
        )
        graph.llm_client.get_langchain_model = Mock(return_value=mock_llm)

        state: AgentState = {
            "messages": [sample_human_message],
            "retrieval_attempts": 1,
            "original_query": sample_human_message.content,
        }

        result = await graph.rewrite_query(state)

        assert isinstance(result["messages"][0], HumanMessage)
        assert result["rewritten_query"]


class TestGenerateAnswerNode:
    @pytest.mark.asyncio
    async def test_generate_answer_success(self, graph, sample_human_message, sample_tool_message):
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(
            return_value=Mock(content="Based on the papers, transformers are neural network architectures.")
        )
        graph.llm_client.get_langchain_model = Mock(return_value=mock_llm)

        state: AgentState = {
            "messages": [sample_human_message, sample_tool_message],
            "retrieval_attempts": 1,
        }

        result = await graph.generate_answer(state)

        assert isinstance(result["messages"][0], AIMessage)
        assert len(result["messages"][0].content) > 0


class TestOutOfScopeNode:
    @pytest.mark.asyncio
    async def test_out_of_scope_response(self, graph, sample_human_message):
        state: AgentState = {
            "messages": [sample_human_message],
            "retrieval_attempts": 0,
        }

        result = await graph.out_of_scope(state)

        assert isinstance(result["messages"][0], AIMessage)


class TestNodeUtils:
    def test_get_latest_query(self, sample_human_message, sample_ai_message):
        query = get_latest_query([sample_human_message, sample_ai_message])
        assert query == "What are attention mechanisms in transformers?"

    def test_get_latest_query_with_multiple_human_messages(self):
        messages = [
            HumanMessage(content="First query"),
            AIMessage(content="First response"),
            HumanMessage(content="Second query"),
        ]
        assert get_latest_query(messages) == "Second query"

    def test_get_latest_context(self, sample_tool_message):
        context = get_latest_context([HumanMessage(content="Query"), sample_tool_message])
        assert "Transformers" in context

    def test_get_latest_context_no_tool_messages(self, sample_human_message):
        assert get_latest_context([sample_human_message]) == ""
