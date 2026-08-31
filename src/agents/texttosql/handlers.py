import logging

from langchain_core.messages import AIMessage

from src.platform.graph import Command, END, MessagesState, NodeError

logger = logging.getLogger(__name__)


async def text_to_sql_error_handler(state: MessagesState, error: NodeError) -> Command:
    """Return a user-facing message when the text-to-SQL workflow cannot complete."""
    logger.error("Text-to-SQL node '%s' failed after retries: %s", error.node, error.error)

    return Command(
        update={
            "messages": [
                AIMessage(
                    content=(
                        "I was unable to query the database due to a temporary error. "
                        "Please try again in a few moments."
                    )
                )
            ],
        },
        goto=END,
    )
