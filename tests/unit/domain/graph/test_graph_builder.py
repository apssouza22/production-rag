"""Tests for GraphBuilder and StateGraphCompiled."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.domain.graph import END, GraphBuilder, MessagesState, START


@pytest.mark.asyncio
async def test_graph_builder_compiles_and_runs_linear_flow() -> None:
    async def echo(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content=f"echo: {state['messages'][-1].content}")]}

    graph = (
        GraphBuilder(MessagesState)
        .add_node("echo", echo)
        .add_edge(START, "echo")
        .add_edge("echo", END)
        .compile()
    )

    result = await graph.ainvoke({"messages": [HumanMessage(content="hello")]})

    assert result["messages"][-1].content == "echo: hello"


@pytest.mark.asyncio
async def test_graph_builder_supports_conditional_edges() -> None:
    async def route(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content="routed")]}

    def pick_next(state: MessagesState) -> str:
        return "route" if state["messages"][-1].content == "go" else END

    graph = (
        GraphBuilder(MessagesState)
        .add_node("route", route)
        .add_edge(START, "route")
        .add_conditional_edges("route", pick_next)
        .compile()
    )

    result = await graph.ainvoke({"messages": [HumanMessage(content="go")]})
    assert result["messages"][-1].content == "routed"


def test_graph_builder_chaining_returns_self() -> None:
    builder = GraphBuilder(MessagesState)
    assert builder.add_node("noop", lambda state: state) is builder
    assert builder.add_edge(START, "noop") is builder
    assert builder.set_node_defaults(retry_policy=None) is builder
