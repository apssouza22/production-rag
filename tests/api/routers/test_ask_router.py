import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock

from src.main import app
from src.agents.knowledgerouter.service import KnowledgeRouterService
from src.agents.knowledgerouter.schemas import AgentResultItem, ClassificationItem
from src import dependencies


@pytest.fixture
def mock_knowledge_router_service():
    service = Mock(spec=KnowledgeRouterService)
    service.ask = AsyncMock(
        return_value={
            "query": "How many transformer papers exist and what do they explain?",
            "answer": "There are 12 transformer papers. They explain self-attention mechanisms.",
            "classifications": [
                ClassificationItem(source="database", query="How many transformer papers exist?"),
                ClassificationItem(source="documents", query="What do transformer papers explain?"),
            ],
            "agent_results": [
                AgentResultItem(
                    source="database",
                    result="There are 12 transformer papers.",
                    metadata={"sql_queries": ["SELECT COUNT(*) FROM papers"]},
                ),
                AgentResultItem(
                    source="documents",
                    result="Transformer papers explain self-attention.",
                    metadata={"sources": ["https://arxiv.org/pdf/1706.03762.pdf"]},
                ),
            ],
            "reasoning_steps": [
                "Classified query into 2 source(s): database, documents",
                "Queried 2 agents in parallel",
                "Synthesized combined answer",
            ],
            "execution_time": 4.5,
            "trace_id": "trace-123",
        }
    )
    return service


@pytest.fixture
def client(mock_knowledge_router_service):
    def override_get_knowledge_router_service():
        return mock_knowledge_router_service

    app.dependency_overrides[dependencies.get_knowledge_router_service] = override_get_knowledge_router_service

    yield TestClient(app)

    app.dependency_overrides.clear()


class TestKnowledgeRouterEndpoint:
    def test_ask_router_success(self, client, mock_knowledge_router_service):
        response = client.post(
            "/api/v1/ask-router",
            json={
                "query": "How many transformer papers exist and what do they explain?",
                "model": "gpt-4o-mini",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["query"] == "How many transformer papers exist and what do they explain?"
        assert "12 transformer papers" in data["answer"]
        assert len(data["classifications"]) == 2
        assert len(data["agent_results"]) == 2
        assert len(data["reasoning_steps"]) == 3
        assert data["trace_id"] == "trace-123"

    def test_ask_router_minimal_request(self, client):
        response = client.post(
            "/api/v1/ask-router",
            json={"query": "What is machine learning?"},
        )

        assert response.status_code == 200
        assert "answer" in response.json()

    def test_ask_router_empty_query(self, client, mock_knowledge_router_service):
        mock_knowledge_router_service.ask = AsyncMock(side_effect=ValueError("Query cannot be empty"))

        response = client.post(
            "/api/v1/ask-router",
            json={"query": ""},
        )

        assert response.status_code == 422

    def test_ask_router_service_error(self, client, mock_knowledge_router_service):
        mock_knowledge_router_service.ask = AsyncMock(side_effect=Exception("Router failed"))

        response = client.post(
            "/api/v1/ask-router",
            json={"query": "Test query"},
        )

        assert response.status_code == 500
        assert "detail" in response.json()
