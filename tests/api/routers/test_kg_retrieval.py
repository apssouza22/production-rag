from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import kg_retrieval
from src.config import get_settings
from src.domain.kg.schemas import DualLevelKGResult


@pytest.fixture
def kg_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(kg_retrieval.router, prefix="/api/v1")
    settings = get_settings()
    app.state.settings = settings
    app.state.kg_retrieval_service = None
    return app


def test_kg_retrieval_debug_disabled_returns_503(kg_test_app: FastAPI) -> None:
    disabled_kg = kg_test_app.state.settings.kg.model_copy(update={"enabled": False})
    kg_test_app.state.settings = kg_test_app.state.settings.model_copy(update={"kg": disabled_kg})

    with TestClient(kg_test_app) as client:
        response = client.post(
            "/api/v1/kg-retrieval/debug",
            json={"query": "What is attention?"},
        )

    assert response.status_code == 503
    assert "disabled" in response.json()["detail"].lower()


def test_kg_retrieval_debug_enabled_returns_payload(kg_test_app: FastAPI) -> None:
    enabled_kg = kg_test_app.state.settings.kg.model_copy(update={"enabled": True})
    kg_test_app.state.settings = kg_test_app.state.settings.model_copy(update={"kg": enabled_kg})

    mock_service = AsyncMock()
    mock_service.retrieve.return_value = DualLevelKGResult(
        query="test",
        high_level_keywords=["theme"],
        low_level_keywords=["entity"],
        formatted_context="context",
    )
    kg_test_app.state.kg_retrieval_service = mock_service

    with TestClient(kg_test_app) as client:
        response = client.post(
            "/api/v1/kg-retrieval/debug",
            json={"query": "test query"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "test"
    assert body["formatted_context"] == "context"
    mock_service.retrieve.assert_awaited_once()
