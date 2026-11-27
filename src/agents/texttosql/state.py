from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class TextToSQLState(TypedDict):
    """State for the text-to-SQL LangGraph workflow."""

    messages: Annotated[list[AnyMessage], add_messages]
    query_loop_count: int