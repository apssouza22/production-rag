from langgraph.graph import StateGraph
from langgraph.types import Checkpointer
from langgraph.typing import ContextT, InputT, NodeInputT, OutputT, StateT

from src.domain.graph.compiled import StateGraphCompiled


class GraphBuilder:
    """Builds and compiles the LangGraph workflow."""

    def __init__(
        self,
        state_schema: type[StateT],
        context_schema: type[ContextT] | None = None,
    ) -> None:
        self.graph = StateGraph(state_schema, context_schema)

    def compile(self, name: str, checkpointer: Checkpointer = None) -> StateGraphCompiled:
        return StateGraphCompiled(self.graph.compile(name=name, checkpointer=checkpointer))
