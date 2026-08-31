"""Tests for LangGraph ToolNode middleware hooks."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import ToolMessage

from src.domain.middleware.pipeline import MiddlewareManager
from src.domain.middleware.tool_hooks import middleware_tool_wrappers
from src.domain.middleware.types import AgentContext, AgentMiddleware


class RecordingMiddleware(AgentMiddleware):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def before_tool_call(
        self,
        ctx: AgentContext,
        *,
        tool_name: str,
        tool_args: dict,
    ) -> dict:
        self.calls.append((tool_name, dict(tool_args)))
        if tool_name == "mutate_args":
            return {**tool_args, "value": tool_args["value"] + 1}
        return tool_args


@pytest.fixture
def middleware_manager() -> tuple[MiddlewareManager, RecordingMiddleware]:
    recording = RecordingMiddleware()
    manager = MiddlewareManager([recording])
    return manager, recording


@pytest.fixture
def agent_context() -> AgentContext:
    return AgentContext(
        messages=[],
        session_id="session-1",
        user_id=None,
        config={},
        agent_name="test-agent",
    )


def _make_request(tool_name: str, tool_args: dict) -> MagicMock:
    request = MagicMock()
    request.tool_call = {"name": tool_name, "args": tool_args, "id": "call-1"}
    request.override.return_value = request
    return request


@pytest.mark.asyncio
async def test_awrap_tool_call_runs_before_tool_call_hooks(
    middleware_manager: tuple[MiddlewareManager, RecordingMiddleware],
    agent_context: AgentContext,
) -> None:
    manager, recording = middleware_manager
    wrappers = middleware_tool_wrappers(manager)
    manager.set_active_ctx(agent_context)

    request = _make_request("search", {"query": "hello"})
    execute = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="call-1"))

    await wrappers["awrap_tool_call"](request, execute)

    assert recording.calls == [("search", {"query": "hello"})]
    execute.assert_awaited_once_with(request)


@pytest.mark.asyncio
async def test_awrap_tool_call_applies_modified_tool_args(
    middleware_manager: tuple[MiddlewareManager, RecordingMiddleware],
    agent_context: AgentContext,
) -> None:
    manager, recording = middleware_manager
    wrappers = middleware_tool_wrappers(manager)
    manager.set_active_ctx(agent_context)

    request = _make_request("mutate_args", {"value": 1})
    overridden_request = MagicMock()
    request.override.return_value = overridden_request
    execute = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="call-1"))

    await wrappers["awrap_tool_call"](request, execute)

    request.override.assert_called_once_with(
        tool_call={"name": "mutate_args", "args": {"value": 2}, "id": "call-1"}
    )
    execute.assert_awaited_once_with(overridden_request)


@pytest.mark.asyncio
async def test_awrap_tool_call_skips_hooks_without_active_context(
    middleware_manager: tuple[MiddlewareManager, RecordingMiddleware],
) -> None:
    manager, recording = middleware_manager
    wrappers = middleware_tool_wrappers(manager)

    request = _make_request("search", {"query": "hello"})
    execute = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="call-1"))

    await wrappers["awrap_tool_call"](request, execute)

    assert recording.calls == []
    execute.assert_awaited_once_with(request)


def test_middleware_tool_wrappers_returns_empty_dict_without_manager() -> None:
    assert middleware_tool_wrappers(None) == {}
