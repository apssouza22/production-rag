"""Pydantic schemas for graph execution trajectories."""

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class TrajectoryEventResponse(BaseModel):
    event_type: str
    name: str
    run_id: str
    parent_run_id: Optional[str] = None
    timestamp: float
    duration_ms: Optional[float] = None
    input: Optional[Any] = None
    output: Optional[Any] = None
    error: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectorySummaryResponse(BaseModel):
    event_count: int
    nodes: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    models: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class GraphTrajectoryResponse(BaseModel):
    started_at: float
    finished_at: Optional[float] = None
    duration_ms: Optional[float] = None
    events: List[TrajectoryEventResponse] = Field(default_factory=list)
    summary: TrajectorySummaryResponse
    steps: List[str] = Field(default_factory=list, description="Human-readable trajectory steps")
