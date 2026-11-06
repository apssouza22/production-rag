"""Read-only subgraph export for KG visualization (Postgres ``kg_nodes`` / ``kg_edges``)."""

import logging
from typing import Annotated

from fastapi import APIRouter, Query

from src.api.kg_visualization_schemas import KgGraphEdgeOut, KgGraphNodeOut, KgGraphResponse
from src.dependencies import SessionDep, SettingsDep
from src.domain.kg.factory import resolve_kg_ingestion_name
from src.domain.kg.models import KgEdge, KgNode
from src.domain.kg.repository import KgGraphRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kg", tags=["kg-visualization"])

_DEFAULT_MAX_NODES = 300
_DEFAULT_MAX_EDGES = 1000
_CAP_MAX_NODES = 500
_CAP_MAX_EDGES = 5000


def _edge_label(edge: KgEdge) -> str:
    desc = (edge.description or "").strip()
    if desc:
        return desc[:160] + ("…" if len(desc) > 160 else "")
    kw = (edge.keywords or "").strip()
    if kw:
        return kw[:160] + ("…" if len(kw) > 160 else "")
    return ""


def _build_response(
    *,
    ingestion_name: str,
    nodes: list[KgNode],
    edges: list[KgEdge],
    max_nodes: int,
    max_edges: int,
) -> KgGraphResponse:
    node_out = [
        KgGraphNodeOut(
            id=n.entity_id,
            label=n.entity_name,
            entity_type=n.entity_type or "UNKNOWN",
            description=(n.description or "")[:2000],
        )
        for n in nodes
    ]
    edge_out = [
        KgGraphEdgeOut(
            source=e.source_entity_id,
            target=e.target_entity_id,
            label=_edge_label(e),
            weight=float(e.weight),
        )
        for e in edges
    ]
    truncated_nodes = len(nodes) >= max_nodes
    truncated_edges = len(edges) >= max_edges
    return KgGraphResponse(
        ingestion_name=ingestion_name,
        nodes=node_out,
        edges=edge_out,
        truncated_nodes=truncated_nodes,
        truncated_edges=truncated_edges,
    )


@router.get("/graph", response_model=KgGraphResponse)
async def get_kg_graph(
    settings: SettingsDep,
    session: SessionDep,
    ingestion_name: Annotated[
        str | None, Query(description="Partition key; defaults to resolved KG/OpenSearch ingestion.")
    ] = None,
    arxiv_id: Annotated[str | None, Query(description="Keep only nodes mentioning this arXiv id.")] = None,
    max_nodes: Annotated[int, Query(ge=1, le=_CAP_MAX_NODES)] = _DEFAULT_MAX_NODES,
    max_edges: Annotated[int, Query(ge=1, le=_CAP_MAX_EDGES)] = _DEFAULT_MAX_EDGES,
) -> KgGraphResponse:
    """Export a bounded subgraph for the web visualizer (does not require ``KG__ENABLED``)."""
    resolved = (ingestion_name or "").strip() or resolve_kg_ingestion_name(settings)
    repo = KgGraphRepository(session, resolved)
    nodes, edges = repo.get_subgraph_for_visualization(
        arxiv_id=arxiv_id,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )
    logger.info(
        "KG graph export: ingestion=%s nodes=%d edges=%d arxiv_filter=%s",
        resolved,
        len(nodes),
        len(edges),
        arxiv_id,
    )
    return _build_response(
        ingestion_name=resolved,
        nodes=nodes,
        edges=edges,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )


__all__ = ["router"]
