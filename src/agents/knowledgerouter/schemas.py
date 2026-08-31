from typing import List, Optional

from pydantic import BaseModel, Field

from src.platform.tracing.schemas import GraphTrajectoryResponse

from .state import Classification, KnowledgeSource


class ClassificationItem(BaseModel):
    """Routing decision for a single knowledge source."""

    source: KnowledgeSource = Field(..., description="Knowledge source to query")
    query: str = Field(..., description="Targeted sub-question for that source")


class AgentResultItem(BaseModel):
    """Result returned by a single sub-agent."""

    source: KnowledgeSource = Field(..., description="Knowledge source that produced this result")
    result: str = Field(..., description="Answer from the sub-agent")
    metadata: dict = Field(default_factory=dict, description="Source-specific metadata")


class KnowledgeRouterRequest(BaseModel):
    """Request model for the knowledge router."""

    query: str = Field(..., description="Natural-language question", min_length=1, max_length=1000)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How many transformer papers are in the database and what do they explain?",
            }
        }


class KnowledgeRouterResponse(BaseModel):
    """Response model for the knowledge router."""

    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Synthesized answer from all consulted sources")
    classifications: List[ClassificationItem] = Field(
        default_factory=list,
        description="Routing decisions made for each knowledge source",
    )
    agent_results: List[AgentResultItem] = Field(
        default_factory=list,
        description="Raw results from each consulted sub-agent",
    )
    reasoning_steps: List[str] = Field(default_factory=list, description="Router workflow steps")
    execution_time: float = Field(..., description="Total execution time in seconds")
    trace_id: Optional[str] = Field(None, description="Langfuse trace ID when tracing is enabled")
    trajectory: Optional[GraphTrajectoryResponse] = Field(
        None,
        description="Full LangGraph execution trajectory captured during the request",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How many transformer papers are in the database and what do they explain?",
                "answer": "There are 12 transformer papers in the database. They primarily cover...",
                "classifications": [
                    {"source": "database", "query": "How many papers mention transformers?"},
                    {"source": "documents", "query": "What do transformer papers explain?"},
                ],
                "agent_results": [
                    {
                        "source": "database",
                        "result": "There are 12 papers with 'transformer' in the title or abstract.",
                        "metadata": {"sql_queries": ["SELECT COUNT(*) FROM papers WHERE ..."]},
                    },
                    {
                        "source": "documents",
                        "result": "Transformer papers explain self-attention mechanisms...",
                        "metadata": {"sources": ["https://arxiv.org/pdf/1706.03762.pdf"]},
                    },
                ],
                "reasoning_steps": [
                    "Classified query into database and documents sources",
                    "Queried 2 agents in parallel",
                    "Synthesized combined answer",
                ],
                "execution_time": 5.2,
                "trace_id": "abc123-def456-ghi789",
                "trajectory": {
                    "started_at": 1712345678.1,
                    "finished_at": 1712345680.4,
                    "duration_ms": 2300,
                    "events": [],
                    "summary": {
                        "event_count": 0,
                        "nodes": [],
                        "tools": [],
                        "models": [],
                        "errors": [],
                    },
                    "steps": ["node:classify", "node:route_agents"],
                },
            }
        }


def classification_to_schema(item: Classification) -> ClassificationItem:
    """Convert a graph classification to an API schema item."""
    return ClassificationItem(source=item["source"], query=item["query"])
