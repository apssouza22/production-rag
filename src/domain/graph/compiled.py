from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from langgraph.typing import ContextT, InputT,  OutputT, StateT

class StateGraphCompiled:
    """Compiled LangGraph workflow ready to run."""

    def __init__(self, state_graph: CompiledStateGraph[StateT, ContextT, InputT, OutputT]):
        self.state_graph = state_graph

    def invoke(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        context: ContextT | None = None
    ):
        """Run the compiled workflow synchronously."""
        return self.state_graph.invoke(input, config, context=context)

    def ainvoke(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        context: ContextT | None = None,
    ):
        """Run the compiled workflow asynchronously."""
        return self.state_graph.ainvoke(input, config, context=context)
