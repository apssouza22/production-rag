import logging
from typing import TYPE_CHECKING

from langgraph.errors import NodeError
from langgraph.types import Command

from src.domain.agent_fault_tolerance.metadata import fault_metadata

if TYPE_CHECKING:
    from src.domain.agents.fusionsearch.state import AgentState

logger = logging.getLogger(__name__)


async def route_agentic_rag_failure(state: "AgentState", error: NodeError) -> Command:
    """Route exhausted Agentic RAG node failures to the graceful fallback node."""
    fault = fault_metadata(error)
    logger.error("Agentic RAG node '%s' failed after retries: %s", error.node, error.error)

    metadata = dict(state.get("metadata") or {})
    metadata["fault_tolerance"] = fault

    return Command(update={"metadata": metadata}, goto="handle_failure")
