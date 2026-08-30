"""Compiled graph ready for synchronous and asynchronous execution."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.typing import ContextT, InputT, OutputT, StateT

from src.domain.graph.types import Command


class StateGraphCompiled:
    """Compiled workflow ready to run."""

    def __init__(
        self,
        state_graph: CompiledStateGraph[StateT, ContextT, InputT, OutputT],
    ) -> None:
        self._state_graph = state_graph

    @property
    def state_graph(self) -> CompiledStateGraph[StateT, ContextT, InputT, OutputT]:
        """Underlying LangGraph compiled graph (for advanced use)."""
        return self._state_graph

    def invoke(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
    ):
        """Run the compiled workflow synchronously."""
        return self._state_graph.invoke(input, config, context=context)

    async def ainvoke(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
    ):
        """Run the compiled workflow asynchronously."""
        return await self._state_graph.ainvoke(input, config, context=context)

    def get_graph(self):
        """Return the graph structure for visualization or introspection."""
        return self._state_graph.get_graph()
