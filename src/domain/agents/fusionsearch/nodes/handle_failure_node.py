import logging
from typing import Dict, List

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from src.domain.agents.fusionsearch.context import Context
from src.domain.agents.fusionsearch.state import AgentState

logger = logging.getLogger(__name__)


async def ainvoke_handle_failure_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, List[AIMessage]]:
    """Return a graceful fallback message after a node exhausts its retries."""
    logger.info("NODE: handle_failure")

    fault = (state.get("metadata") or {}).get("fault_tolerance", {})
    failed_node = fault.get("failed_node", "unknown")

    response_text = (
        "I apologize, but I encountered a temporary issue while processing your request "
        f"at the '{failed_node}' step.\n\n"
        "This is usually caused by a brief network or service interruption. "
        "Please try your question again in a few moments.\n\n"
        "If the problem persists, the upstream LLM or search service may be unavailable."
    )

    return {"messages": [AIMessage(content=response_text)]}
