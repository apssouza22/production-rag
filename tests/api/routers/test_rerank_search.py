from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from src import dependencies
from src.domain.search.schemas import DocumentSearchResult, SearchDocument
from src.main import app


@pytest.fixture
def mock_rerank_search_service():
    service = Mock()
    service.config = Mock(use_hybrid=True, rerank_enabled=True)

    async def _search(*, query: str, top_k: int) -> DocumentSearchResult:
        return DocumentSearchResult(
            query=query,
            search_mode="hybrid",
            rerank_applied=True,
            before_rerank=[
                SearchDocument(
                    arxiv_id="1111.11111",
                    chunk_text="first chunk",
                    title="Paper One",
                    score=0.9,
                    rank=0,
                ),
                SearchDocument(
                    arxiv_id="2222.22222",
                    chunk_text="second chunk",
                    title="Paper Two",
                    score=0.8,
                    rank=1,
                ),
            ],
            after_rerank=[
                SearchDocument(
                    arxiv_id="2222.22222",
                    chunk_text="second chunk",
                    title="Paper Two",
                    score=0.98,
                    rank=0,
                    rerank_score=0.98,
                    original_rank=1,
                ),
                SearchDocument(
                    arxiv_id="1111.11111",
                    chunk_text="first chunk",
                    title="Paper One",
                    score=0.75,
                    rank=1,
                    rerank_score=0.75,
                    original_rank=0,
                ),
            ],
        )

    service.search_rrf = AsyncMock(side_effect=_search)
    return service


@pytest.fixture
def mock_paper_rrf_search():
    client = Mock()
    client.health_check.return_value = True
    return client


@pytest.fixture
def client(mock_rerank_search_service, mock_paper_rrf_search):
    def override_rerank_search_service():
        return mock_rerank_search_service

    def override_paper_rrf_search():
        return mock_paper_rrf_search

    app.dependency_overrides[dependencies.get_rerank_search_service] = override_rerank_search_service
    app.dependency_overrides[dependencies.get_hybrid_rrf_search] = override_paper_rrf_search

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_rerank_search_endpoint_success(client, mock_rerank_search_service):
    response = client.post(
        "/api/v1/rerank-search",
        json={
            "query": "machine learning",
            "top_k": 2,
            "use_hybrid": True,
            "rerank_enabled": True,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "machine learning"
    assert data["search_mode"] == "hybrid"
    assert data["rerank_applied"] is True
    assert len(data["before_rerank"]) == 2
    assert len(data["after_rerank"]) == 2
    assert data["after_rerank"][0]["arxiv_id"] == "2222.22222"
    assert data["after_rerank"][0]["rerank_score"] == 0.98

    mock_rerank_search_service.search_rrf.assert_awaited_once_with(query="machine learning", top_k=2)


def test_rerank_search_endpoint_minimal_request(client, mock_rerank_search_service):
    response = client.post("/api/v1/rerank-search", json={})

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "machine learning neural networks"
    mock_rerank_search_service.search_rrf.assert_awaited_once_with(
        query="machine learning neural networks",
        top_k=10,
    )


def test_rerank_search_endpoint_validation_errors(client):
    response = client.post("/api/v1/rerank-search", json={"query": ""})
    assert response.status_code == 422

    response = client.post("/api/v1/rerank-search", json={"query": "test", "top_k": 0})
    assert response.status_code == 422


def test_rerank_search_endpoint_unavailable(client, mock_paper_rrf_search):
    mock_paper_rrf_search.health_check.return_value = False

    response = client.post("/api/v1/rerank-search", json={"query": "machine learning"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Search service is currently unavailable"
