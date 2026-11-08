from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api import kg_visualization
from src.dependencies import get_db_session
from src.domain.kg.models import KgEdge, KgNode


@pytest.fixture
def kg_viz_app() -> FastAPI:
    app = FastAPI()
    app.include_router(kg_visualization.router, prefix="/api/v1")

    class _Settings:
        kg = MagicMock()
        opensearch = MagicMock()

    app.state.settings = _Settings()
    app.state.settings.kg.ingestion_name = ""
    app.state.settings.opensearch.ingestion_name = "eval-default"

    def _fake_session():
        yield MagicMock()

    app.dependency_overrides[get_db_session] = _fake_session
    return app


def _node(ingestion: str) -> KgNode:
    return KgNode(
        id=uuid4(),
        entity_id="attention",
        entity_name="Attention",
        entity_type="METHOD",
        description="Scaled dot-product attention.",
        source_chunk_keys=["1706.03762:3"],
        document_ids=["1706.03762"],
        ingestion_name=ingestion,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _edge(ingestion: str) -> KgEdge:
    return KgEdge(
        id=uuid4(),
        source_entity_id="attention",
        target_entity_id="transformer",
        description="used in",
        keywords="architecture",
        weight=1.0,
        source_chunk_keys=["1706.03762:3"],
        ingestion_name=ingestion,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def test_kg_graph_returns_payload(kg_viz_app: FastAPI) -> None:
    ingestion = "eval-default"
    nodes = [_node(ingestion)]
    edges = [_edge(ingestion)]

    with patch.object(kg_visualization.KgGraphRepository, "get_subgraph_for_visualization", return_value=(nodes, edges)):
        with TestClient(kg_viz_app) as client:
            response = client.get("/api/v1/kg/graph")

    assert response.status_code == 200
    body = response.json()
    assert body["ingestion_name"] == ingestion
    assert len(body["nodes"]) == 1
    assert body["nodes"][0]["id"] == "attention"
    assert body["nodes"][0]["label"] == "Attention"
    assert len(body["edges"]) == 1
    assert body["edges"][0]["source"] == "attention"
    assert body["edges"][0]["target"] == "transformer"
    assert "used in" in body["edges"][0]["label"]
