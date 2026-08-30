"""Graph builder that wraps LangGraph behind a single domain API."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Any, Self

from langgraph.graph import StateGraph
from langgraph.typing import ContextT, StateT

from src.domain.graph.compiled import StateGraphCompiled
from src.domain.graph.types import Checkpointer


class GraphBuilder:
    """Build and compile agent workflows behind a single domain API."""

    def __init__(
        self,
        state_schema: type[StateT],
        context_schema: type[ContextT] | None = None,
    ) -> None:
        self._graph = StateGraph(state_schema, context_schema)

    def set_node_defaults(self, **kwargs: Any) -> Self:
        self._graph.set_node_defaults(**kwargs)
        return self

    def add_node(self, node: str, action: Any | None = None, /, **kwargs: Any) -> Self:
        self._graph.add_node(node, action, **kwargs)
        return self

    def add_edge(self, start_key: str | Any, end_key: str | Any) -> Self:
        self._graph.add_edge(start_key, end_key)
        return self

    def add_conditional_edges(
        self,
        source: str,
        path: Callable[..., Hashable | Sequence[Hashable]],
        path_map: dict[Hashable, str] | list[str] | None = None,
    ) -> Self:
        self._graph.add_conditional_edges(source, path, path_map)
        return self

    def compile(
        self,
        *,
        name: str | None = None,
        checkpointer: Checkpointer = None,
        **kwargs: Any,
    ) -> StateGraphCompiled:
        return StateGraphCompiled(self._graph.compile(name=name, checkpointer=checkpointer, **kwargs))
