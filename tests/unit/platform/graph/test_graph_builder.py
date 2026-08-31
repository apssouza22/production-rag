"""Tests for GraphBuilder and StateGraphCompiled."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src.platform.graph import END, GraphBuilder, MessagesState, START, ToolNode
from src.platform.middleware.pipeline import MiddlewareManager
from src.platform.middleware.types import AgentContext, AgentMiddleware


class RecordingMiddleware(AgentMiddleware):
    def __init__(self) -> None:
        self.before_calls: list[tuple[str, dict]] = []
        self.after_calls: list[tuple[str, str]] = []

    async def before_tool_call(
        self,
        ctx: AgentContext,
        *,
        tool_name: str,
        tool_args: dict,
    ) -> dict:
        self.before_calls.append((tool_name, dict(tool_args)))
        return tool_args

    async def after_tool_call(
        self,
        ctx: AgentContext,
        *,
        tool_name: str,
        tool_result: Any,
    ) -> Any:
        self.after_calls.append((tool_name, tool_result.content))
        return tool_result


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
    assert builder.set_middleware_manager(MiddlewareManager([])) is builder


@tool
def echo_tool(value: str) -> str:
    """Echo the provided value."""
    return value


@pytest.mark.asyncio
async def test_add_tool_node_applies_middleware_wrappers() -> None:
    recording = RecordingMiddleware()
    manager = MiddlewareManager([recording])
    manager.set_active_ctx(
        AgentContext(
            messages=[],
            session_id="session-1",
            user_id=None,
            config={},
            agent_name="test-agent",
        )
    )

    graph = (
        GraphBuilder(MessagesState)
        .set_middleware_manager(manager)
        .add_tool_node("tools", [echo_tool])
        .add_edge(START, "tools")
        .add_edge("tools", END)
        .compile()
    )

    result = await graph.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "echo_tool",
                            "args": {"value": "hello"},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }
    )

    assert recording.before_calls == [("echo_tool", {"value": "hello"})]
    assert recording.after_calls == [("echo_tool", "hello")]
    assert result["messages"][-1].content == "hello"


@pytest.mark.asyncio
async def test_add_node_applies_middleware_to_existing_tool_node() -> None:
    recording = RecordingMiddleware()
    manager = MiddlewareManager([recording])
    manager.set_active_ctx(
        AgentContext(
            messages=[],
            session_id="session-1",
            user_id=None,
            config={},
            agent_name="test-agent",
        )
    )
    tool_node = ToolNode([echo_tool], name="tools")

    graph = (
        GraphBuilder(MessagesState)
        .set_middleware_manager(manager)
        .add_node("tools", tool_node)
        .add_edge(START, "tools")
        .add_edge("tools", END)
        .compile()
    )

    await graph.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "echo_tool",
                            "args": {"value": "hello"},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }
    )

    assert recording.before_calls == [("echo_tool", {"value": "hello"})]
    assert recording.after_calls == [("echo_tool", "hello")]
    assert tool_node._wrap_tool_call is not None
    assert tool_node._awrap_tool_call is not None
