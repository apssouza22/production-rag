"""Tests for AgenticRAGGraph node methods."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from src.agents.fusionsearch.config import GraphConfig
from src.agents.fusionsearch.graph import AgenticRAGGraph
from src.agents.fusionsearch.models import AnswerJudgement, GradeDocuments, SourceItem
from src.agents.fusionsearch.state import AgentState
from src.agents.fusionsearch.utils import get_latest_context, get_latest_query


@pytest.fixture
def graph(mock_rerank_search_service):
    """AgenticRAGGraph with mocked dependencies."""
    config = GraphConfig(
        model="gpt-4o-mini",
        temperature=0.0,
        max_retrieval_attempts=2,
        guardrail_threshold=60,
    )
    return AgenticRAGGraph(
        rerank_search_service=mock_rerank_search_service,
        config=config,
    )


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
        assert "don't have enough information" in content_lower
        assert result["abstained"] is True


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
        state: AgentState = {
            "messages": [sample_human_message, sample_tool_message],
            "retrieval_attempts": 1,
        }

        with patch("src.agents.fusionsearch.graph.make_chat_model", return_value=mock_llm):
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
        state: AgentState = {
            "messages": [sample_human_message, sample_tool_message],
            "retrieval_attempts": 1,
        }

        with patch("src.agents.fusionsearch.graph.make_chat_model", return_value=mock_llm):
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
        state: AgentState = {
            "messages": [sample_human_message],
            "retrieval_attempts": 1,
            "original_query": sample_human_message.content,
        }

        with patch("src.agents.fusionsearch.graph.make_chat_model", return_value=mock_llm):
            result = await graph.rewrite_query(state)

        assert isinstance(result["messages"][0], HumanMessage)
        assert result["rewritten_query"]


class TestAssessRetrievalNode:
    @pytest.mark.asyncio
    async def test_assess_retrieval_passes_confident_results(self, graph, sample_human_message):
        import json

        from langchain_core.messages import ToolMessage

        payload = {
            "formatted_context": "Transformers paper content",
            "retrieval_confidence": 0.82,
            "document_count": 2,
            "sources": [
                {
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "authors": ["Vaswani"],
                    "url": "https://arxiv.org/pdf/1706.03762.pdf",
                    "relevance_score": 0.82,
                }
            ],
        }
        state: AgentState = {
            "messages": [
                sample_human_message,
                ToolMessage(content=json.dumps(payload), tool_call_id="retrieve_1", name="retrieve_papers"),
            ],
            "retrieval_attempts": 1,
        }

        result = await graph.check_retrieval_score(state)

        assert result["routing_decision"] == "grade_documents"
        assert result["retrieval_confidence"] == 0.82
        assert len(result["relevant_sources"]) == 1

    @pytest.mark.asyncio
    async def test_assess_retrieval_abstains_on_low_confidence(self, graph, sample_human_message):
        import json

        from langchain_core.messages import ToolMessage

        payload = {
            "formatted_context": "",
            "retrieval_confidence": 0.1,
            "document_count": 0,
            "sources": [],
        }
        state: AgentState = {
            "messages": [
                sample_human_message,
                ToolMessage(content=json.dumps(payload), tool_call_id="retrieve_1", name="retrieve_papers"),
            ],
            "retrieval_attempts": 1,
        }

        result = await graph.check_retrieval_score(state)

        assert result["routing_decision"] == "abstain"
        assert result["abstained"] is True


class TestValidateAnswerNode:
    @pytest.mark.asyncio
    async def test_validate_answer_abstains_on_invalid_citations(self, graph, sample_human_message):
        sources = [
            SourceItem(
                arxiv_id="1706.03762",
                title="Attention Is All You Need",
                authors=["Vaswani"],
                url="https://arxiv.org/pdf/1706.03762.pdf",
                relevance_score=0.9,
            )
        ]
        state: AgentState = {
            "messages": [
                sample_human_message,
                AIMessage(content="Unsupported claim [Source: Unknown Paper]."),
            ],
            "relevant_sources": sources,
        }

        result = await graph.validate_answer(state)

        assert result["abstained"] is True
        assert result["response_status"] == "abstained_citation"

    @pytest.mark.asyncio
    async def test_validate_answer_passes_with_supported_judgement(self, graph, sample_human_message):
        sources = [
            SourceItem(
                arxiv_id="1706.03762",
                title="Attention Is All You Need",
                authors=["Vaswani"],
                url="https://arxiv.org/pdf/1706.03762.pdf",
                relevance_score=0.9,
            )
        ]
        mock_structured_llm = Mock()
        mock_structured_llm.ainvoke = AsyncMock(
            return_value=AnswerJudgement(
                verdict="supported",
                confidence=0.92,
                reasoning="Answer is grounded in the retrieved context",
            )
        )
        mock_llm = Mock()
        mock_llm.with_structured_output = Mock(return_value=mock_structured_llm)
        state: AgentState = {
            "messages": [
                sample_human_message,
                AIMessage(content="Transformers use attention [Source: Attention Is All You Need]."),
            ],
            "relevant_sources": sources,
        }

        with patch("src.agents.fusionsearch.graph.make_chat_model", return_value=mock_llm):
            result = await graph.validate_answer(state)

        assert result["abstained"] is False
        assert result["response_status"] == "answered"


class TestGenerateAnswerNode:
    @pytest.mark.asyncio
    async def test_generate_answer_success(self, graph, sample_human_message, sample_tool_message):
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(
            return_value=Mock(content="Based on the papers, transformers are neural network architectures.")
        )
        state: AgentState = {
            "messages": [sample_human_message, sample_tool_message],
            "retrieval_attempts": 1,
        }

        with patch("src.agents.fusionsearch.graph.make_chat_model", return_value=mock_llm):
            result = await graph.generate_answer(state)

        assert isinstance(result["messages"][0], AIMessage)
        assert len(result["messages"][0].content) > 0

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
