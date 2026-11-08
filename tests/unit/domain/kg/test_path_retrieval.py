"""PathRAG path discovery on fixture graphs (not wired through DB)."""

from datetime import datetime, timezone

from src.domain.kg.models import KgEdge, KgNode
from src.domain.kg.path_retrieval import PathDiscover, cap_subgraph_edges_by_nodes


def _node(entity_id: str, name: str) -> KgNode:
    now = datetime.now(timezone.utc)
    return KgNode(
        entity_id=entity_id,
        entity_name=name,
        entity_type="CONCEPT",
        description=f"Description of {name}",
        source_chunk_keys=[],
        document_ids=[],
        ingestion_name="test",
        created_at=now,
        updated_at=now,
    )


def _edge(src: str, tgt: str, keywords: str = "relates") -> KgEdge:
    now = datetime.now(timezone.utc)
    return KgEdge(
        source_entity_id=src,
        target_entity_id=tgt,
        description="",
        keywords=keywords,
        weight=1.0,
        source_chunk_keys=[],
        ingestion_name="test",
        created_at=now,
        updated_at=now,
    )


def test_discover_finds_two_hop_path_between_seeds() -> None:
    """Connected chain a-b-c yields at least one path with hop_count >= 2 for seeds a,c."""
    nodes = {
        "a": _node("a", "Alpha"),
        "b": _node("b", "Beta"),
        "c": _node("c", "Gamma"),
    }
    edges = [_edge("a", "b", "ab"), _edge("b", "c", "bc")]
    discoverer = PathDiscover(enabled=True, max_hops=3, max_paths=15)
    hits = discoverer.discover(["a", "c"], nodes, edges)
    assert hits
    assert any(hit.hop_count >= 2 for hit in hits)
    assert any("Alpha" in hit.narrative for hit in hits)
    assert any("Gamma" in hit.narrative for hit in hits)


def test_discover_returns_empty_when_seeds_disconnected() -> None:
    nodes = {"a": _node("a", "Alpha"), "b": _node("b", "Beta")}
    edges = [_edge("a", "x"), _edge("b", "y")]
    hits = PathDiscover(enabled=True, max_hops=3).discover(["a", "b"], nodes, edges)
    assert hits == []


def test_discover_requires_two_seeds() -> None:
    nodes = {"a": _node("a", "Alpha")}
    hits = PathDiscover(enabled=True).discover(["a"], nodes, [_edge("a", "b")])
    assert hits == []


def test_cap_subgraph_edges_respects_max_nodes() -> None:
    edges = [
        _edge("a", "b", "ab"),
        _edge("b", "c", "bc"),
        _edge("c", "d", "cd"),
    ]
    capped = cap_subgraph_edges_by_nodes(edges, ["a", "d"], max_nodes=3)
    assert capped == [edges[0]]
    endpoints: set[str] = {"a", "d"}
    for edge in capped:
        endpoints |= {edge.source_entity_id, edge.target_entity_id}
    assert len(endpoints) <= 3


def test_cap_subgraph_edges_keeps_all_when_under_budget() -> None:
    edges = [_edge("a", "b"), _edge("b", "c")]
    assert cap_subgraph_edges_by_nodes(edges, ["a", "c"], max_nodes=10) == edges


def test_discover_returns_empty_when_disabled() -> None:
    nodes = {"a": _node("a", "Alpha"), "c": _node("c", "Gamma")}
    edges = [_edge("a", "c")]
    hits = PathDiscover(enabled=False).discover(["a", "c"], nodes, edges)
    assert hits == []
