from unittest.mock import MagicMock

import pytest
from src.domain.kg.models import KgNode
from src.domain.kg.repository import KgGraphRepository


@pytest.fixture
def session() -> MagicMock:
    return MagicMock()


def test_upsert_node_creates_new(session: MagicMock) -> None:
    session.scalar.return_value = None
    repo = KgGraphRepository(session, ingestion_name="dev")
    entity_id = repo.upsert_node(
        entity_name="Transformer",
        entity_type="MODEL",
        description="An architecture.",
        chunk_key="1234.5678:0",
        arxiv_id="1234.5678",
    )
    assert entity_id == "transformer"
    session.add.assert_called_once()
    added = session.add.call_args[0][0]
    assert isinstance(added, KgNode)
    assert added.source_chunk_keys == ["1234.5678:0"]


def test_node_degrees_aggregates_source_and_target(session: MagicMock) -> None:
    repo = KgGraphRepository(session, ingestion_name="dev")
    session.execute.side_effect = [
        [("entity_a", 2)],
        [("entity_a", 1)],
    ]
    degrees = repo.node_counts(["entity_a"])
    assert degrees["entity_a"] == 3
