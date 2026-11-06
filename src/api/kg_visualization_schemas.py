from pydantic import BaseModel, Field


class KgGraphNodeOut(BaseModel):
    """One entity node for vis-network (``id`` is the stable ``entity_id`` slug)."""

    id: str
    label: str
    entity_type: str
    description: str = ""


class KgGraphEdgeOut(BaseModel):
    """Directed edge between two entity ids."""

    source: str = Field(description="Source entity_id")
    target: str = Field(description="Target entity_id")
    label: str = Field(default="", description="Short relation label for the graph")
    weight: float = 1.0


class KgGraphResponse(BaseModel):
    """Subgraph for visualization (bounded by ``max_nodes`` / ``max_edges``)."""

    ingestion_name: str
    nodes: list[KgGraphNodeOut]
    edges: list[KgGraphEdgeOut]
    truncated_nodes: bool = Field(
        default=False,
        description="True when there may be more nodes than returned (hit max_nodes).",
    )
    truncated_edges: bool = Field(
        default=False,
        description="True when there may be more edges than returned (hit max_edges).",
    )


__all__ = ["KgGraphEdgeOut", "KgGraphNodeOut", "KgGraphResponse"]
