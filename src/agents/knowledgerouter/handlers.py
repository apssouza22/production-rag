import logging
from typing import TYPE_CHECKING

from src.domain.graph import Command, END, NodeError

from src.domain.graph.metadata import fault_metadata

if TYPE_CHECKING:
    from src.agents.knowledgerouter.state import RouterState

logger = logging.getLogger(__name__)


async def knowledge_router_error_handler(state: "RouterState", error: NodeError) -> Command:
    """Recover from knowledge-router failures with partial results when possible."""
    fault = fault_metadata(error)
    logger.error("Knowledge router node '%s' failed after retries: %s", error.node, error.error)

    if error.node == "classify":
        return Command(
            update={
                "classifications": [{"source": "documents", "query": state["query"]}],
            },
            goto="documents",
        )

    if error.node in ("documents", "database"):
        return Command(
            update={
                "results": [
                    {
                        "source": error.node,
                        "result": (
                            "This knowledge source is temporarily unavailable. "
                            "Please try again in a few moments."
                        ),
                        "metadata": fault,
                    }
                ],
            },
            goto="synthesize",
        )

    if error.node == "synthesize":
        partial = "\n\n".join(
            f"**From {item['source'].title()}:**\n{item['result']}"
            for item in state.get("results", [])
        )
        answer = partial or "Unable to generate an answer due to a temporary service error."
        return Command(update={"final_answer": answer}, goto=END)

    return Command(
        update={"final_answer": "An unexpected error occurred. Please try again later."},
        goto=END,
    )
