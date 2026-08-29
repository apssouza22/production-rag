"""Tests for the text-to-SQL API endpoint."""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from src import dependencies
from src.agents.texttosql.service import TextToSQLService
from src.main import app


@pytest.fixture
def mock_text_to_sql_service():
    service = Mock(spec=TextToSQLService)
    service.ask = AsyncMock(
        return_value={
            "query": "How many papers?",
            "answer": "There are 10 papers.",
            "sql_queries": ["SELECT COUNT(*) FROM papers"],
            "reasoning_steps": ["Listed available database tables"],
            "execution_time": 1.23,
            "trace_id": None,
        }
    )
    return service


@pytest.fixture
def client(mock_text_to_sql_service):
    def override_get_text_to_sql_service():
        return mock_text_to_sql_service

    app.dependency_overrides[dependencies.get_text_to_sql_service] = override_get_text_to_sql_service
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_ask_sql_endpoint(client, mock_text_to_sql_service):
    response = client.post(
        "/api/v1/ask-sql",
        json={"query": "How many papers?", "model": "gpt-4o-mini"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "There are 10 papers."
    assert data["sql_queries"] == ["SELECT COUNT(*) FROM papers"]
    mock_text_to_sql_service.ask.assert_called_once()


def test_ask_sql_empty_query(client, mock_text_to_sql_service):
    mock_text_to_sql_service.ask = AsyncMock(side_effect=ValueError("Query cannot be empty"))

    response = client.post(
        "/api/v1/ask-sql",
        json={"query": "", "model": "gpt-4o-mini"},
    )

    assert response.status_code == 422
