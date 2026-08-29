"""Tests for the knowledge router agent."""

from unittest.mock import AsyncMock, Mock

import pytest

from src.agents.knowledgerouter.config import KnowledgeRouterConfig
from src.agents.knowledgerouter.schemas import AgentResultItem, ClassificationItem
from src.agents.knowledgerouter.service import KnowledgeRouterService


@pytest.fixture
def mock_agentic_rag():
    service = Mock()
    service.ask = AsyncMock(
        return_value={
            "query": "What is a transformer?",
            "answer": "Transformers use self-attention mechanisms.",
            "sources": [{"url": "https://arxiv.org/pdf/1706.03762.pdf"}],
            "reasoning_steps": ["Retrieved papers", "Generated answer"],
            "retrieval_attempts": 1,
        }
    )
    return service


@pytest.fixture
def mock_text_to_sql():
    service = Mock()
    service.ask = AsyncMock(
        return_value={
            "query": "How many papers?",
            "answer": "There are 42 papers.",
            "sql_queries": ["SELECT COUNT(*) FROM papers"],
            "reasoning_steps": ["Listed tables", "Executed SQL"],
        }
    )
    return service


@pytest.fixture
def mock_ollama_client():
    client = Mock()
    client.get_langchain_model.return_value = Mock()
    return client


@pytest.fixture
def router_service(mock_agentic_rag, mock_text_to_sql, mock_ollama_client, monkeypatch):
    monkeypatch.setattr(
        "src.domain.agents.knowledgerouter.service.build_knowledge_router_graph",
        lambda **kwargs: AsyncMock(),
    )

    return KnowledgeRouterService(
        agentic_rag_service=mock_agentic_rag,
        text_to_sql_service=mock_text_to_sql,
        llm_client=mock_ollama_client,
        langfuse_tracer=None,
        agent_config=KnowledgeRouterConfig(model="gpt-4o-mini"),
    )


class TestKnowledgeRouterService:
    def test_service_initialization(self, router_service):
        assert router_service.agentic_rag is not None
        assert router_service.text_to_sql is not None
        assert router_service.graph is not None

    @pytest.mark.asyncio
    async def test_ask_empty_query_validation(self, router_service):
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await router_service.ask(query="")

    @pytest.mark.asyncio
    async def test_ask_returns_structured_response(self, router_service):
        router_service.graph.ainvoke = AsyncMock(
            return_value={
                "query": "How many transformer papers exist and what do they explain?",
                "classifications": [
                    {"source": "database", "query": "How many transformer papers exist?"},
                    {"source": "documents", "query": "What do transformer papers explain?"},
                ],
                "results": [
                    {
                        "source": "database",
                        "result": "There are 12 transformer papers.",
                        "metadata": {"sql_queries": ["SELECT COUNT(*) FROM papers WHERE title ILIKE '%transformer%'"]},
                    },
                    {
                        "source": "documents",
                        "result": "Transformer papers explain self-attention.",
                        "metadata": {"sources": ["https://arxiv.org/pdf/1706.03762.pdf"]},
                    },
                ],
                "final_answer": "There are 12 transformer papers. They explain self-attention.",
            }
        )

        result = await router_service.ask(query="How many transformer papers exist and what do they explain?")

        assert result["query"] == "How many transformer papers exist and what do they explain?"
        assert "12 transformer papers" in result["answer"]
        assert len(result["classifications"]) == 2
        assert len(result["agent_results"]) == 2
        assert len(result["reasoning_steps"]) >= 2

    def test_extract_reasoning_steps_single_source(self, router_service):
        classifications = [ClassificationItem(source="documents", query="What is BERT?")]
        agent_results = [
            AgentResultItem(source="documents", result="BERT is a language model.", metadata={}),
        ]

        steps = router_service._extract_reasoning_steps(classifications, agent_results)

        assert "documents" in steps[0]
        assert any("documents agent" in step for step in steps)

    def test_extract_reasoning_steps_multiple_sources(self, router_service):
        classifications = [
            ClassificationItem(source="database", query="Count papers"),
            ClassificationItem(source="documents", query="Explain papers"),
        ]
        agent_results = [
            AgentResultItem(source="database", result="10 papers", metadata={}),
            AgentResultItem(source="documents", result="Papers explain AI", metadata={}),
        ]

        steps = router_service._extract_reasoning_steps(classifications, agent_results)

        assert any("parallel" in step for step in steps)
        assert any("Synthesized" in step for step in steps)
