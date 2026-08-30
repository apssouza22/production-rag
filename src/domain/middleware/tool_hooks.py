"""LangGraph ToolNode hooks that dispatch to the agent middleware pipeline."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from .pipeline import MiddlewareManager

ExecuteToolCall = Callable[..., Any]


def middleware_tool_wrappers(
    manager: Optional[MiddlewareManager],
) -> dict[str, Any]:
    """Return ``wrap_tool_call`` / ``awrap_tool_call`` kwargs for ``ToolNode``.

    When no manager is provided, returns an empty dict so ``ToolNode`` behaves
    as usual. The wrappers read ``manager.active_ctx``, which is set for the
    duration of ``AgentPipeline.run()``.
    """
    if manager is None:
        return {}

    return {
        "wrap_tool_call": _build_sync_wrapper(manager),
        "awrap_tool_call": _build_async_wrapper(manager),
    }


def _apply_before_tool_call(
    manager: MiddlewareManager,
    request: Any,
) -> Any:
    ctx = manager.active_ctx
    if ctx is None:
        return request

    tool_call = request.tool_call
    modified_args = asyncio.run(
        manager.run_before_tool_call(
            ctx,
            tool_name=tool_call["name"],
            tool_args=tool_call["args"],
        )
    )
    if modified_args == tool_call["args"]:
        return request

    return request.override(tool_call={**tool_call, "args": modified_args})


async def _aapply_before_tool_call(
    manager: MiddlewareManager,
    request: Any,
) -> Any:
    ctx = manager.active_ctx
    if ctx is None:
        return request

    tool_call = request.tool_call
    modified_args = await manager.run_before_tool_call(
        ctx,
        tool_name=tool_call["name"],
        tool_args=tool_call["args"],
    )
    if modified_args == tool_call["args"]:
        return request

    return request.override(tool_call={**tool_call, "args": modified_args})


def _build_sync_wrapper(manager: MiddlewareManager) -> ExecuteToolCall:
    def wrap_tool_call(request: Any, execute: ExecuteToolCall) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            request = _apply_before_tool_call(manager, request)
        return execute(request)

    return wrap_tool_call


def _build_async_wrapper(manager: MiddlewareManager) -> ExecuteToolCall:
    async def awrap_tool_call(request: Any, execute: ExecuteToolCall) -> Any:
        request = await _aapply_before_tool_call(manager, request)
        return await execute(request)

    return awrap_tool_call
