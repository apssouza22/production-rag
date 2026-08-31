"""Composable middleware for agent invocations.

Usage::

    from src.platform.middleware import (
        AgentContext,
        AgentPipeline,
        ErrorHandlingMiddleware,
        LoggingMiddleware,
    )

    pipeline = AgentPipeline(
        middlewares=[LoggingMiddleware(), ErrorHandlingMiddleware()],
        invoke_fn=agent.core_invoke,
    )
    result = await pipeline.run(ctx)
"""

from .error_handling_middleware import ErrorHandlingMiddleware
from .logging_middleware import LoggingMiddleware
from .pipeline import AgentPipeline, MiddlewareManager
from .tool_hooks import middleware_tool_wrappers
from .trajectory_middleware import TrajectoryMiddleware
from .types import AgentContext, AgentMiddleware, InvokeResult, NextFn, build_invoke_config

__all__ = [
    "AgentContext",
    "AgentMiddleware",
    "AgentPipeline",
    "ErrorHandlingMiddleware",
    "InvokeResult",
    "LoggingMiddleware",
    "MiddlewareManager",
    "NextFn",
    "TrajectoryMiddleware",
    "build_invoke_config",
    "middleware_tool_wrappers",
]
